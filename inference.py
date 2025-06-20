import json
import torch
import torchaudio
import accelerate
import argparse
import os

from generation_utils import load_model, process_batch

MODEL_PATH = "fnlp/MOSS-TTSD-v0"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

def main():
    parser = argparse.ArgumentParser(description="TTS inference with Asteroid model")
    parser.add_argument("--jsonl", default="examples/examples.jsonl", 
                       help="Path to JSONL file (default: examples/examples.jsonl)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_dir", default="outputs",
                       help="Output directory for generated audio files (default: outputs)")
    parser.add_argument("--use_normalize", action="store_true", default=True,
                       help="Whether to use text normalization (default: True)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    tokenizer, model, spt = load_model(MODEL_PATH, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device)
    model = model.to(device)
    
    # Load the items from the JSONL file
    try:
        with open(args.jsonl, "r") as f:
            items = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(items)} items from {args.jsonl}")
    except FileNotFoundError:
        print(f"Error: JSONL file '{args.jsonl}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        return
    
    # Fix the seed for reproducibility
    accelerate.utils.set_seed(args.seed)
    print(f"Set random seed to {args.seed}")
    
    # Process the batch of items
    print("Starting inference...")
    actual_texts_data, audio_results = process_batch(
        batch_items=items,
        tokenizer=tokenizer,
        model=model,
        spt=spt,
        device=device,
        system_prompt=SYSTEM_PROMPT,
        start_idx=0,
        use_normalize=args.use_normalize
    )
    
    # Save the audio results to files
    saved_count = 0
    for idx, audio_result in enumerate(audio_results):
        if audio_result is not None:
            output_path = os.path.join(args.output_dir, f"output_{idx}.wav")
            torchaudio.save(
                output_path,
                audio_result["audio_data"],
                audio_result["sample_rate"]
            )
            print(f"Saved audio to {output_path}")
            saved_count += 1
        else:
            print(f"Skipping sample {idx} due to generation error")
    
    print(f"Inference completed. Saved {saved_count}/{len(items)} audio files to {args.output_dir}")

if __name__ == "__main__":
    main()