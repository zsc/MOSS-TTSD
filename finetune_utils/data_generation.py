import json
import torch
import numpy as np
import argparse
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation_utils import process_jsonl_item, normalize_text, load_audio_data
from transformers import AutoTokenizer
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8
SILENCE_DURATION = 0.0  # Fixed silence duration: 0 seconds

def load_tokenizer(model_path, spt_config_path, spt_checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    spt = XY_Tokenizer.load_from_checkpoint(config_path=spt_config_path, ckpt_path=spt_checkpoint_path)
    spt.eval()
    return tokenizer, spt

def process_inputs(tokenizer, spt, prompt, text, device, audio_data=None, max_channels=8, pad_token=1024):
    seq = f"<|begin_of_style|>{prompt}<|end_of_style|>\n<|begin_of_text|>{text}<|end_of_text|>\n<|begin_of_speech|>"
    inputs1 = np.array(tokenizer.encode(seq))
    input_ids = np.full((inputs1.shape[0], max_channels), pad_token)
    input_ids[:, 0] = inputs1

    if audio_data is not None:
        try:
            # audio_data should now be a processed audio tensor
            wav = audio_data

            # Add fixed 5-second silence at the end of audio (using 16k sample rate)
            silence_samples = int(SILENCE_DURATION * 16000)
            silence = torch.zeros(wav.shape[0], silence_samples)
            wav = torch.cat([wav, silence], dim=1)

            with torch.no_grad():
                # Use SPT encoding
                encode_result = spt.encode([wav.squeeze().to(device)])
                audio_token = encode_result["codes_list"][0].permute(1, 0).cpu().numpy()  # Adjust dimension order

            # similar to DAC encoding adjustment
            audio_token[:, 0] = audio_token[:, 0] + 151665  # Keep this line if offset is needed, otherwise delete
            input_ids = np.concatenate([input_ids, audio_token])
        except Exception as e:
            print(f"Error processing audio data: {e}")
            raise

    return input_ids

def process_data(
        jsonl: str ,
        model_path: str ,
        output_dir: str ,
        data_name: str = "processd_data",
        use_normalize: bool = True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    tokenizer, spt = load_tokenizer(model_path, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device)
    
    # Load the items from the JSONL file
    try:
        with open(jsonl, "r") as f:
            items = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(items)} items from {jsonl}")
    except FileNotFoundError:
        print(f"Error: JSONL file '{jsonl}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        return

    # 用于存储所有处理后的数据
    all_data = []
    offsets = []
    
    for idx, item in enumerate(items):
        # Use new processing function
        processed_item = process_jsonl_item(item)
        
        text = processed_item["text"]
        prompt_text = processed_item["prompt_text"]
        
        # Merge text
        full_text = prompt_text + text
        
        # Apply text normalization based on parameter
        if use_normalize:
            full_text = normalize_text(full_text)
        
        # Replace speaker tags
        final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")
        audio_data = load_audio_data(processed_item["prompt_audio"]) if processed_item["prompt_audio"] else None
        input_id = process_inputs(tokenizer, spt, SYSTEM_PROMPT, final_text, device, audio_data, max_channels=MAX_CHANNELS)
        
        # 创建数据条目，只包含input_ids
        data_entry = {
            'input_ids': input_id  # shape: [seq_len, 8]
        }
        
        all_data.append(data_entry)

        print(f"Processed item {idx + 1}/{len(items)}: input_ids shape {input_id.shape}")
    
    # 保存pkl文件 - 逐个序列化
    output_pkl_path = os.path.join(output_dir, f"{data_name}.pkl")
    with open(output_pkl_path, 'wb') as f:
        for data_entry in all_data:
            offsets.append(f.tell())  # 记录当前位置作为偏移量
            pickle.dump(data_entry, f)  # 单独序列化每个数据条目
    
    # 保存offset元数据文件
    output_meta_path = os.path.join(output_dir, f"{data_name}_metas.npy")
    np.save(output_meta_path, np.array(offsets))  # 注意这里需要包装成二维数组
    
    print(f"Saved {len(all_data)} processed items to {output_pkl_path}")
    print(f"Saved offset metadata to {output_meta_path}")
    print(f"Total sequences processed: {len(all_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS inference with Asteroid model")
    parser.add_argument("--jsonl", type=str, required=True,
                       help="Path to JSONL file")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for generated audio files")
    parser.add_argument("--data_name", default="processed_data",
                       help="Name of the processed data file (default: processed_data)")
    parser.add_argument("--use_normalize", action="store_true", default=True,
                       help="Whether to use text normalization (default: True)")
    
    args = parser.parse_args()
    
    if not args.jsonl:
        raise ValueError("JSONL file path is required.")
    elif not os.path.exists(args.jsonl):
        raise ValueError(f"JSONL file '{args.jsonl}' does not exist.")
    if not args.model_path:
        args.model_path = MODEL_PATH
    elif not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist.")
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    process_data(args.jsonl, args.model_path, args.output_dir, args.data_name, args.use_normalize)
