<div align="center">
    <h1>
    MOSS: Text to Spoken Dialogue Generation
    </h1>
    <p>
    <img src="asset/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
    <p>
    </p>
    <a href="https://www.open-moss.com/en/moss-ttsd/"><img src="https://img.shields.io/badge/Blog-Read%20More-green" alt="blog"></a>
    <a href="https://www.open-moss.com/en/moss-ttsd/"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-orange" alt="paper"></a>
    <a href="https://huggingface.co/fnlp/MOSS-TTSD-v0.5"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://huggingface.co/spaces/fnlp/MOSS-TTSD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"></a>
    <a href="https://github.com/"><img src="https://img.shields.io/badge/Python-3.10+-orange" alt="version"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/PyTorch-2.0+-brightgreen" alt="python"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>

# MOSS-TTSD 🪐

## Overview

MOSS-TTSD (text to spoken dialogue) is an open-source bilingual spoken dialogue synthesis model that supports both Chinese and English.
It can transform dialogue scripts between two speakers into natural, expressive conversational speech.
MOSS-TTSD supports voice cloning and long single-session speech generation, making it ideal for AI podcast production.
 For detailed information about the model and demos, please refer to our [Blog-en](https://www.open-moss.com/en/moss-ttsd/) and [中文博客](https://www.open-moss.com/cn/moss-ttsd/). You can also find the model on [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) and try it out in the [Spaces demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Highlights

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation.
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.

## News 🚀

- **[2025-07-04]** MOSS-TTSD v0.5 is released! v0.5 has enhanced the accuracy of timbre switching, voice cloning capability, and model stability. We recommend using the v0.5 model by default.
- **[2025-06-20]** MOSS-TTSD v0 is released! Moreover, we provide a podcast generation pipeline named Podever, which can automatically convert PDF, URL, or long text files into high-quality podcasts.

## Installation

To run MOSS-TTSD, you need to install the required dependencies. You can use either pip or conda to set up your environment.

### Using conda

```bash
conda create -n moss_ttsd python=3.10 -y && conda activate moss_ttsd
pip install -r requirements.txt
pip install flash-attn
```

### Download XY Tokenizer

<!-- https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/main/xy_tokenizer.ckpt 下载到 Asteroid-gradio/XY_Tokenizer/weights -->

You also need to download the XY Tokenizer model weights. You can find the weights in the [XY_Tokenizer repository](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0).

```bash
mkdir -p XY_Tokenizer/weights
huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/
```

## Usage

### Local Inference

To run MOSS-TTSD locally, you can use the provided inference script. Make sure you have the model checkpoint and configuration files ready.

```bash
python inference.py --jsonl examples/examples.jsonl --output_dir outputs --seed 42 --use_normalize
```

Parameters:

- `--jsonl`: Path to the input JSONL file containing dialogue scripts and speaker prompts.
- `--output_dir`: Directory where the generated audio files will be saved.
- `--seed`: Random seed for reproducibility.
- `--use_normalize`: Whether to normalize the text input (default is `True`).
- `--dtype`: Model data type (default is `bf16`).
- `--attn_implementation`: Attention implementation (default is `flash_attention_2`, `sdpa` and `eager` are also supported).

#### JSONL Input Format

The input JSONL file should contain one JSON object per line with the following structure:

```json
{
  "base_path": "examples",
  "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]...",
  "prompt_audio_speaker1": "path/to/speaker1_audio.wav",
  "prompt_text_speaker1": "Reference text for speaker 1 voice cloning",
  "prompt_audio_speaker2": "path/to/speaker2_audio.wav", 
  "prompt_text_speaker2": "Reference text for speaker 2 voice cloning"
}
```

Field descriptions:

- `base_path`: Base directory path for relative file paths
- `text`: Dialogue script with speaker tags `[S1]` and `[S2]` indicating speaker turns
- `prompt_audio_speaker1/2`: Path to reference audio files for voice cloning (relative to `base_path`)
- `prompt_text_speaker1/2`: Reference text corresponding to the audio prompts for better voice matching

In addition to the JSONL format above, the system also supports using a single JSON object where both speakers share the same reference audio file:

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]...",
  "prompt_audio": "path/to/shared_reference_audio.wav",
  "prompt_text": "[S1]Reference text for speaker 1[S2]Reference text for speaker 2"
}
```

Field descriptions:

- `base_path`: Base directory path for audio files
- `text`: Dialogue script with speaker tags `[S1]` and `[S2]` indicating speaker turns
- `prompt_audio`: Path to shared reference audio file containing both speakers' voices (relative to `base_path`)
- `prompt_text`: Reference text corresponding to the audio, also using `[S1]` and `[S2]` tags to distinguish speakers

The dialogue text uses speaker tags to indicate turns:

- `[S1]`: Indicates Speaker 1 is speaking
- `[S2]`: Indicates Speaker 2 is speaking

Example:

```
[S1]Hello, how are you today?[S2]I'm doing great, thanks for asking![S1]That's wonderful to hear.
```

**GPU Requirements**

Our model is efficient and has low VRAM requirements.

For example, when generating 600 seconds of audio at the default bf16 precision, the model uses less than 7GB of VRAM. This should make it compatible with most consumer-grade GPUs. You can estimate the VRAM needed for a specific audio length using this formula:

$$
y = 0.00172x + 5.8832
$$

Here, $x$ is the desired length of your audio in seconds, and $y$ is the estimated VRAM cost in GB.

> Please note that if your own prompts (e.g., `prompt_audio_speaker1`) are longer than our default examples, VRAM usage will be higher.

| Length of the Generated Audio(Second) | GPU Memory Cost(GB) |
| ------------------------------------- | ------------------- |
| 120                                   | 6.08                |
| 300                                   | 6.39                |
| 360                                   | 6.5                 |
| 600                                   | 6.91                |

### Web UI Usage

You can run the MOSS-TTSD web UI locally using Gradio. Run the following command to start the Gradio demo:

```bash
python gradio_demo.py
```

### API Usage

Powered by siliconflow. Stay tuned!

### Podcast Generation

We provide a podcast generation tool that directly analyzes either a URL or a user-uploaded PDF file, extracting content to generate a high-quality podcast segment.

Before using the podcast generation tool, please ensure that environment variables `OPENAI_API_KEY` and `OPENAI_API_BASE` are set correctly.
We use Gemini API to generate the podcast script.
So the API key should be set to the Gemini API key and the API base should be set to "https://generativelanguage.googleapis.com/v1beta/openai/"

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="your_openai_api_base"

# Process web article
python podcast_generate.py "https://www.open-moss.com/cn/moss-ttsd/"

# Process PDF file  
python podcast_generate.py "examples/Attention Is All You Need.pdf"

# Process text file
python podcast_generate.py "examples/example.txt"

# Custom output directory
python podcast_generate.py "your_input" -o "your_output"

# Generate a podcast in English
python podcast_generate.py "your_input" -l en
```

The tool supports generating scripts in both English (`en`) and Chinese (`zh`), defaulting to Chinese. You can use the `--language` or `-l` flag to specify the language.

## Fine-Tuning

We provide basic fine-tuning scripts and tools for preprocessing the required fine-tuning data, which are located in the `finetune_utils` folder.

### File Structure

```
MOSS-TTSD/
├── finetune_workflow.py              # One-click fine-tuning workflow script
└── finetune_utils/
    ├── data_preprocess.py            # Data preprocessing script
    ├── finetune.py                   # Fine-tuning training script
    ├── training_config.yaml          # Training configuration template
    └── finetune_config.yaml          # Workflow configuration template
```

### Environment Setup

Before running fine-tuning scripts, please make sure you have installed all required dependencies. You can use the following commands to set up the environment:

#### Using conda

```bash
conda create -n moss_ttsd_finetune python=3.10 -y && conda activate moss_ttsd_finetune
pip install -r requirements_finetune.txt
pip install flash-attn
```

#### Using venv

```bash
python -m venv moss_ttsd_finetune
source moss_ttsd_finetune/bin/activate
pip install -r requirements_finetune.txt
pip install flash-attn --no-build-isolation
```

### Data Preparation

Following the data organization format described in the previous section [Usage/Local Inference/JSONL Input Format](#jsonl-input-format) create your JSONL files. Each file can contain one or more entries that conform to the specified format. You can refer to examples.jsonl and examples_single_reference.jsonl in the examples folder for guidance.

Once you have prepared the JSONL file, you can manually preprocess the data using the `data_preprocess.py` tool. For example:

```bash
python finetune_utils/data_preprocess.py --jsonl <path_to_jsonl> --model_path <path_to_model> --output_dir <output_directory> --data_name <data_name> [--use_normalize]
```

> **⚠️ Important**: For better stability and to avoid path resolution issues, we strongly recommend using absolute paths for all file and directory parameters instead of relative paths.

#### Parameters

- `--jsonl`: Path to the JSONL input file (required)
- `--model_path`: Path to the pre-trained MOSS-TTSD model directory
- `--output_dir`: Directory where processed data will be saved (required)
- `--data_name`: Name prefix for the output files (default: `processed_data`)
- `--use_normalize`: Enable text normalization (default: True)

#### Output Files

The script will generate two files in the specified output directory:

1. `<data_name>.pkl`: Contains the processed training data with input_ids
2. `<data_name>_metas.npy`: Contains offset metadata for efficient data loading

### Training

After generating the processed training data, you can use the `finetune.py` script to fine-tune the MOSS-TTSD model on your custom dataset.

#### Usage

```bash
python finetune_utils/finetune.py --model_path <path_to_model> --data_dir <path_to_processed_data> --output_dir <output_directory> [--training_cfg <training_config_file>]
```

> **⚠️ Important**: For better stability and to avoid path resolution issues, we strongly recommend using absolute paths for all file and directory parameters instead of relative paths.

#### Parameters

- `--model_path`: Path to the pre-trained MOSS-TTSD model directory
- `--data_dir`: Directory containing the processed training data (.pkl and _metas.npy files) (required)
- `--output_dir`: Directory where the fine-tuned model will be saved (required)
- `--training_cfg`: Path to the training configuration YAML file (default: `training_config.yaml`)

#### Training Configuration

The training parameters can be configured via a YAML file. The default configuration is located at `finetune_utils/training_config.yaml`.

### One-Click Fine-Tuning Workflow

For a simplified fine-tuning experience, we provide a complete workflow script (`finetune_workflow.py`) that automates both data preprocessing and model fine-tuning in a single command. This eliminates the need to run separate scripts and ensures a streamlined process.

#### Quick Start

1. **Configure your workflow**: Fill in the configuration template at `finetune_utils/finetune_config.yaml`
2. **Run the workflow**: Execute the workflow script with your configuration

#### Configuration Template

The workflow uses a YAML configuration file to specify all parameters. You can find an empty template at `finetune_utils/finetune_config.yaml`:

```yaml
path_to_jsonl :           # Path to the training data in JSONL format
data_output_directory :   # Directory where the processed data will be saved
data_name :               # Name of the dataset   
use_normalize :           # Whether to normalize the data (true/false)
path_to_model :           # Path to the pre-trained model (leave empty to use default HuggingFace model)
finetuned_model_output :  # Directory where the finetuned model will be saved
training_config_file :    # Path to the training configuration file
```

#### Example Configuration

```yaml
path_to_jsonl : /path/to/your/training_data.jsonl
data_output_directory : /path/to/processed_data
data_name : my_dataset
use_normalize : true
path_to_model : # Leave empty to use fnlp/MOSS-TTSD-v0.5 from HuggingFace
finetuned_model_output : /path/to/output/fine_tuned_model
training_config_file : /path/to/training_config.yaml
```

#### Usage

```bash
python finetune_workflow.py --cfg path/to/your/config.yaml [--pass_data_preprocess]
```

> **💡 Tip**: Use absolute paths in the configuration file to avoid path resolution issues.

#### Parameters

- `-c`, `--cfg`: Path to the workflow configuration YAML file (default: `./finetune_utils/finetune_config.yaml`)
- `-pd`, `--pass_data_preprocess`: Skip data preprocess step and proceed directly to fine-tuning

## Demos

See our blog for more demos at https://www.open-moss.com/en/moss-ttsd/

## Limitations

Currently, our model still exhibits instances of instability, such as speaker switching errors and timbre cloning deviations.
We will further optimize the model for stability in subsequent versions.

## License

MOSS-TTSD is released under the Apache 2.0 license.

## Citation

```
@article{moss2025ttsd,
  title={Text to Spoken Dialogue Generation}, 
  author={OpenMOSS Team},
  year={2025}
}
```

## ⚠️ Usage Disclaimer

This project provides an open-source spoken dialogue synthesis model intended for academic research, educational purposes, and legitimate applications such as AI podcast production, assistive technologies, and linguistic research. Users must not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities, and should ensure compliance with local laws and regulations while upholding ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.
