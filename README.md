<div align="center">
    <h1>
    MOSS: Text to Spoken Dialogue Generation
    </h1>
    <p>
    <img src="asset/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
    <p>
    </p>
    <a href="https://blog.example.com"><img src="https://img.shields.io/badge/Blog-Read%20More-green" alt="blog"></a>
    <a href="#"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-orange" alt="paper"></a>
    <a href="https://huggingface.co/fnlp/MOSS-TTSD-v0"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://github.com/"><img src="https://img.shields.io/badge/Python-3.12+-orange" alt="version"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


# MOSS-TTSD 🪐

## Overview

MOSS-TTSD (text to spoken dialogue) is an open-source bilingual spoken dialogue synthesis model that supports both Chinese and English.
It can transform dialogue scripts between two speakers into natural, expressive conversational speech.
MOSS-TTSD supports voice cloning and single-session speech generation of up to 960 seconds, making it ideal for AI podcast production.

## Highlights

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation (up to 960 seconds)**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation, enabling single-session speech generation of up to 960 seconds.
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.


## News 🚀

- **[2025-06-20]** MOSS-TTSD v0 is released!

## Installation

To run MOSS-TTSD, you need to install the required dependencies. You can use either pip or conda to set up your environment.

### Using conda

```bash
conda create -n moss_ttsd python=3.10 -y && conda activate moss_ttsd
pip install -r requirements.txt
```

### Download XY Tokenizer

<!-- https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/main/xy_tokenizer.ckpt 下载到 Asteroid-gradio/XY_Tokenizer/weights -->

You also need to download the XY Tokenizer model weights. You can find the weights in the [XY_Tokenizer repository](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0).

```bash
mkdir -p XY_Tokenizer/weights
wget https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/main/xy_tokenizer.ckpt -O XY_Tokenizer/weights/xy_tokenizer.ckpt
```

## Usage

### Local Inference

To run MOSS-TTSD locally, you can use the provided inference script. Make sure you have the model checkpoint and configuration files ready.

```bash
python inference.py --jsonl examples/examples.jsonl --output_dir output --seed 42 --use_normalize
```

Parameters:
- `--jsonl`: Path to the input JSONL file containing dialogue scripts and speaker prompts.
- `--output_dir`: Directory where the generated audio files will be saved.
- `--seed`: Random seed for reproducibility.
- `--use_normalize`: Whether to normalize the text input (default is `True`).

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

The dialogue text uses speaker tags to indicate turns:
- `[S1]`: Indicates Speaker 1 is speaking
- `[S2]`: Indicates Speaker 2 is speaking

Example:
```
[S1]Hello, how are you today?[S2]I'm doing great, thanks for asking![S1]That's wonderful to hear.
```

**GPU Requirements**

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

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="your_openai_api_base"

# Process web article
python podcast_generate.py "https://www.example.com/article"

# Process PDF file  
python podcast_generate.py "examples/Attention Is All You Need.pdf"

# Process text file
python podcast_generate.py "your_article.txt"

# Custom output directory
python podcast_generate.py "your_input" -o "your_output"
```

## Demos

## Evaluation
| Model      | Speaker Accuracy (↑) | Corrected Speaker Accuracy (↑) | Speaker Similarity (↑) | WER (↓) | Normalized WER (↓) |
|------------|----------------------|--------------------------------|-----------------------------|---------|--------------------|
| Asteroid   | 0.7756               | 0.8506                         | 0.7272                      | 0.2836  | 0.2257             |
| Mooncast   | 0.8240               | 0.8426                         | 0.7267                      | 0.5407  | 0.2631             |


We constructed a test set comprising 240 two-speaker dialogue samples. The Meta's Massively Multilingual Speech Forced Alignment (MMS-FA) model was employed to perform word-level alignment between the input text and the output audio. The output audio was subsequently segmented into sentence-level clips based on punctuation marks, with speaker labels for each segment determined by the input text.

For speaker verification, the wespeaker SimAMResNet100 model was utilized as the speaker embedding extractor. For each audio segment, the cosine similarity of speaker embeddings was computed against the audio samples of the two speakers in the prompt. The predicted speaker for each segment was assigned as the one with the higher similarity score.

The speaker accuracy for each sample was derived by averaging the accuracy across all segments. To account for potential speaker flipping instability during generation, we introduced a corrected speaker accuracy metric. For each sample, the accuracy was calculated twice: once using the original speaker labels and once using the flipped labels. The higher of the two values was taken as the final speaker accuracy for that sample.

The speaker similarity score for each segment was defined as the higher of the two cosine similarity values between the segment and the prompt audio samples of the two speakers. The overall speaker similarity metric for the model was obtained by averaging these values across all segments.

For the Word Error Rate (WER) evaluation, the input text was compared against the ASR-transcribed output of the 240 generated samples. The ASR transcription was performed using the Whisper Large-v3 model. Both the input text and ASR output were normalized by removing all speaker identifiers and punctuation marks. To further mitigate potential ASR-induced errors, the input text and ASR output were converted to pinyin (phonetic transcription), yielding a normalized WER metric.


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