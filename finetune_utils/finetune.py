import os
import torch
import random
import pickle
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_asteroid import AsteroidTTSInstruct
from transformers import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
import argparse

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
MAX_CHANNELS = 8

class LazySupervisedDataset(Dataset):
    def __init__(self, data_dir, channels: int, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer, self.channels = tokenizer, channels
        pkls = [os.path.join(data_dir, each) for each in os.listdir(data_dir) if each.endswith('.pkl')]
        self.data = []
        for pkl_file in pkls:
            metas = np.load(pkl_file.replace(".pkl", "_metas.npy"))
            f = open(pkl_file, "rb")
            for start_pointer in metas:
                f.seek(start_pointer) 
                self.data.append(pickle.load(f))
            f.close()
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def truncate_and_shift(self, example: Dict[str, List]) -> Dict[str, np.ndarray]:
        input_ids = np.array(example["input_ids"])[:, :self.channels]
        labels = np.array(example["input_ids"])[:, :self.channels]
        seq_len = input_ids.shape[0]
        new_seq_len = seq_len + self.channels - 1

        shifted_input_ids = np.full((new_seq_len, self.channels), 1024)
        shifted_input_ids[:, 0] = np.full(new_seq_len, self.tokenizer.pad_token_id)
        shifted_labels = np.full((new_seq_len, self.channels), -100)

        # Delay Pattern: Shift input_ids and labels
        for i in range(self.channels):
            shifted_input_ids[i : (seq_len + i), i] = input_ids[:, i]
            shifted_labels[i : (seq_len + i), i] = labels[:, i]
        
        return {
            "input_ids": shifted_input_ids,
            "labels": shifted_labels,
            "attention_mask": np.ones(new_seq_len)
        }
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        line = self.data[i]
        result = self.truncate_and_shift(line)
        return {
            "input_ids": torch.tensor(result["input_ids"], dtype=torch.long),
            "labels": torch.tensor(result["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(result["attention_mask"], dtype=torch.long)
        }

@dataclass
class DataCollatorForSupervisedDataset:
    pad_token_id: int
    max_length: int
    filler_token_id: int = 1024

    def __call__(self, instances: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        channels = input_ids[0].shape[1]
        max_length = min(max(ids.shape[0] for ids in input_ids), self.max_length)
        padded_input_ids, padded_labels, padded_attns = [], [], []
        
        for ids, lbls, attn in zip(input_ids, labels, attention_masks):
            seq_len = ids.shape[0]
            if seq_len < max_length:
                pad_len = max_length - seq_len
                input_pad = np.full((pad_len, channels), self.filler_token_id)
                input_pad[:, 0] = self.pad_token_id
                padded_input_ids.append(np.concatenate([ids, input_pad]))
                label_pad = np.full((pad_len, channels), -100)
                padded_labels.append(np.concatenate([lbls, label_pad]))
                attn_pad = np.zeros(pad_len)
                padded_attns.append(np.concatenate([attn, attn_pad]))
            else:
                padded_input_ids.append(ids[:max_length])
                padded_labels.append(lbls[:max_length])
                padded_attns.append(attn[:max_length])

        input_ids = torch.tensor(np.stack(padded_input_ids), dtype=torch.long)
        labels = torch.tensor(np.stack(padded_labels), dtype=torch.long)
        attention_mask = torch.tensor(np.stack(padded_attns), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def train(model_path : str, data_dir : str, output_dir : str, training_cfg : Dict, device: str = "cuda"):
    print("初始化 tokenizer 和 model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    model = AsteroidTTSInstruct.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.set_weights([8,2,1,1,1,1,1,1])
    model.config.use_cache = False
    model.to(torch.device(device))
    
    print("初始化 dataloader")
    train_dataset = LazySupervisedDataset(data_dir, MAX_CHANNELS, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, 2000)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg.get('per_device_train_batch_size', 2)),
        gradient_accumulation_steps=int(training_cfg.get('gradient_accumulation_steps', 2)),
        num_train_epochs=int(training_cfg.get('num_train_epochs', 1000)),
        learning_rate=float(training_cfg.get('learning_rate', 1e-4)),
        bf16=bool(training_cfg.get('bf16', True)),
        logging_steps=int(training_cfg.get('logging_steps', 10)),
        save_steps=int(training_cfg.get('save_steps', 12650)),
        save_total_limit=int(training_cfg.get('save_total_limit', 10)),
        dataloader_num_workers=int(training_cfg.get('dataloader_num_workers', 1)),
        warmup_ratio=float(training_cfg.get('warmup_ratio', 0.01)),
        lr_scheduler_type=str(training_cfg.get('lr_scheduler_type', "cosine")),
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs")
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    trainer.train()
    torch.cuda.synchronize()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Asteroid TTS Instruct Model")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated audio files")
    parser.add_argument("--training_cfg", type=str, default=None,
                        help="Path to the training configuration file")
    
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = MODEL_PATH
    elif not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist.")
    if not args.data_dir:
        raise ValueError("Data directory is required.")
    elif not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory '{args.data_dir}' does not exist.")
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    training_cfg = {}
    if args.training_cfg:
        import yaml
        with open(args.training_cfg, 'r') as f:
            training_cfg = yaml.safe_load(f)
    
    train(args.model_path, args.data_dir, args.output_dir, training_cfg, device="cuda" if torch.cuda.is_available() else "cpu")