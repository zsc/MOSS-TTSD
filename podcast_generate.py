import os
import torch
import torchaudio
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader
import openai
from generation_utils import load_model, process_batch
import argparse

# =============== Configuration Section ===============
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
MODEL_PATH = "fnlp/MOSS-TTSD-v0"
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

# Default audio file paths (default audio provided by user)
DEFAULT_PROMPT_AUDIO_SPEAKER1 = "examples/zh_spk1_moon.wav"
DEFAULT_PROMPT_TEXT_SPEAKER1 = "周一到周五，每天早晨七点半到九点半的直播片段。言下之意呢，就是废话有点多，大家也别嫌弃，因为这都是直播间最真实的状态了。"
DEFAULT_PROMPT_AUDIO_SPEAKER2 = "examples/zh_spk2_moon.wav"
DEFAULT_PROMPT_TEXT_SPEAKER2 = "如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起畅聊新消费新科技新趋势。"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============== Text Parsing Functions ===============

def extract_text_from_pdf(file_path):
    """Extract text content from PDF file
    
    Args:
        file_path (str): PDF file path
        
    Returns:
        str: Extracted text content, returns None if failed
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF content extraction failed: {str(e)}")
        return None


def extract_web_content(url):
    """Extract title and main content from web URL
    
    Args:
        url (str): Web URL address
        
    Returns:
        tuple: (title, content) - title and main content, returns None if failed
    """
    try:
        # Send request and get web content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'  # Ensure correct Chinese encoding
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        h1_tags = soup.find_all('h1')
        if h1_tags:
            for h1 in h1_tags:
                if h1.text.strip():
                    title = h1.text.strip()
                    break

        if not title and soup.title:
            title = soup.title.string.strip()

        # Locate main article content
        content_div = None

        # Try to find main content area
        possible_content_divs = soup.find_all('div',
                                              class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower()))
        if possible_content_divs:
            content_div = max(possible_content_divs, key=lambda x: len(x.get_text()))

        # If above method fails, look for areas near author/source information
        if not content_div:
            author_info = soup.find(text=re.compile(r'作者：|来源：'))
            if author_info and author_info.parent:
                # Look up for possible content containers
                parent = author_info.parent
                for _ in range(5):  # Look up at most 5 levels
                    if parent.name == 'div' and len(parent.get_text()) > 500:
                        content_div = parent
                        break
                    parent = parent.parent
                    if not parent:
                        break

        # If still not found, try to find the longest p tag collection
        if not content_div:
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Find paragraph collection with most text
                content_div = max(paragraphs, key=lambda x: len(x.get_text()))

        # Extract main content
        text_content = ""
        if content_div:
            # Remove unwanted elements
            for unwanted in content_div.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                unwanted.decompose()

            # Get cleaned text
            text_content = content_div.get_text(separator='\n', strip=True)
        else:
            # Fallback: get all text from body
            body = soup.find('body')
            if body:
                for unwanted in body.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()
                text_content = body.get_text(separator='\n', strip=True)

        # Clean text
        # Remove extra blank lines
        cleaned_text = re.sub(r'\n{2,}', '\n\n', text_content)
        # Remove extra spaces
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

        return title, cleaned_text

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def extract_text_from_txt(file_path):
    """Extract text content from TXT file
    
    Args:
        file_path (str): TXT file path
        
    Returns:
        str: Extracted text content, returns None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            print(f"TXT file reading failed (tried GBK encoding): {str(e)}")
            return None
    except Exception as e:
        print(f"TXT file reading failed: {str(e)}")
        return None


def parse_input_content(input_path):
    """Parse input content, supports URL, PDF or TXT files
    
    Args:
        input_path (str): URL address, PDF file path or TXT file path
        
    Returns:
        str: Parsed text content
    """
    print(f"Parsing input: {input_path}")
    
    # Check if it's a URL
    if input_path.startswith(('http://', 'https://')):
        print("URL detected, extracting web content...")
        result = extract_web_content(input_path)
        if result:
            title, content = result
            print(f"Web title: {title}")
            print(f"Content length: {len(content)} characters")
            return f"{title}\n\n{content}" if title else content
        else:
            print("Web content extraction failed")
            return None
    
    # Check if it's a PDF file
    elif input_path.lower().endswith('.pdf'):
        print("PDF file detected, extracting content...")
        content = extract_text_from_pdf(input_path)
        if content:
            print(f"PDF content length: {len(content)} characters")
            return content
        else:
            print("PDF content extraction failed")
            return None
    
    # Check if it's a TXT file
    elif input_path.lower().endswith('.txt'):
        print("TXT file detected, reading content...")
        content = extract_text_from_txt(input_path)
        if content:
            print(f"TXT file content length: {len(content)} characters")
            return content
        else:
            print("TXT file reading failed")
            return None
    
    else:
        print(f"Unsupported input format: {input_path}")
        return None


# =============== Dialogue Script Generation Function ===============

def generate_podcast_script(content):
    """Call large model to generate podcast dialogue script"""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "YOUR_API_BASE_URL"),
    )

    
    role_play = "两位中文播客主持人"
    
    instruction = f"""你是一位专业的中文播客文字脚本撰稿人。现在请你根据提供的有关最新AI及大模型相关进展的原始资料，生成一段模拟{role_play}之间的自然对话脚本。该脚本应符合以下具体要求：
    一、语言风格
    - 使用较为自然、随意、轻松的日常中文表达；
    - 优先采用简单易懂的词汇，避免书面用语，将书面表达转换为符合口语表达的形式，但不改变专业词汇的内容；
    - 可适度加入网络流行语、俗语、俚语，增强真实感；
    - 符合{role_play}对话的感觉。

    二、句式结构
    - 使用松散、自然的句式，允许存在口语特征如重复、停顿、语气词等；
    - 鼓励使用叠词（如"特别特别"、"慢慢来"）和填充词（如"这个"、"其实"、"然后"、"就是"、"呃"等）；
    - 可适度插入模糊表达、略带情绪的语调，增强亲和力。

    三、对话结构
    - 两个说话人交替发言，并使用[S1]和[S2]标记两位说话人轮次，[S1]和[S2]中间不加入换行；
    - 每当一方讲话时，另一方可以适当插入自然、简短的反馈或承接语（如"嗯。""对。""是的。""确实。""原来是这样。"等），展现倾听状态；
    - 对话应有开头引入、核心讨论与自然结尾，语气上有节奏起伏，避免平铺直叙；
    - 总长度控制在10分钟以内的语音朗读时长（不超过1500字），禁止超时。
    - **特别强调听话方的积极反馈：当一位说话人正在讲述或解释某个观点时，另一位说话人应频繁地插入简短的承接或反馈词语（例如：「嗯。」、「是。」、「对。」、「哦。」、「是的。」、「哦，原来是这样。」、「明白。」、「没错。」、「有道理。」、「确实」），以表明其正在积极倾听、理解和互动。这些反馈应自然地穿插在说话者语句的间歇或段落转换处，而不是生硬地打断。例如：[S2]我本人其实不太相信星座诶，[S1]嗯。[S2]在一开始的时候，我就跟大部分不相信星座的一样，觉得，呃，你总能把人就分成十二种，[S1]是的。[S2]然后呢就它讲的就是对的。这种反馈要尽可能多，不要吝啬。**

    四、标点与格式
    - 仅使用中文标点：逗号、句号、问号；
    - 禁止使用叹号。禁止使用省略号（'...'）、括号、引号（包括''""'"）或波折号等特殊符号；
    - 所有数字转换为中文表达，如"1000000"修改为"一百万"；
    - 请根据上下文，智慧地判断数字的读音，所有带数字的英文缩写要意译，如"a2b"输出为"a到b"、"gpt-4o"输出为"GPT四O"、"3:4"输出为"3比4"，"2021"如果表达年份，应当转换为"二零二一"，但如果表示数字，应当转换为"两千零二十一"。请保证不要简单转换为中文数字，而是根据上下文，将其翻译成合适的中文。

    五、内容要求
    - 所有内容都基于原始资料改写，不得照搬其书面表达，原始资料中所有的内容都要完整的提到，不能丢失或者省略信息；
    - 可加入适当的背景说明、吐槽、对比、联想、提问等方式，增强对话的节奏和趣味性；
    - 确保信息密度较高，引用需确保上下文完整，确保听众能理解；
    - 在对话内不要输出"我是谁"，"我是S1"，"我是S2"等相关内容；
    - 如有专业术语则需要提供解释的解释，如果涉及抽象技术点，可以使用比喻类比等方式解释，避免听起来晦涩难懂。

    ## 原始资料
    {content}

    请根据以上要求和提供的原始资料，将其转化为符合以上所有要求的播客对话脚本。一定要用[S1]和[S2]标记两位说话人，绝对不能使用任何其它符号标记说话人。
    注意：直接输出结果，不要包含任何额外信息。
    """

    try:
        print("Calling large model to generate dialogue script...")
        print(f"Input content length: {len(content)} characters")
        
        completion = client.chat.completions.create(
            model="gemini-2.5-pro",
            # model="gemini-2.5-flash-preview-04-17",
            messages=[
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        )
        
        raw_result = completion.choices[0].message.content
        
        # Remove all newlines
        processed_result = raw_result.replace('\n', '').replace('\r', '')
        
        print("=" * 50)
        print("Large model generated dialogue script (original version):")
        print("=" * 50)
        print(raw_result)
        print("=" * 50)
        print("Large model generated dialogue script (after removing newlines):")
        print("=" * 50)
        print(processed_result)
        print("=" * 50)
        print(f"Original script length: {len(raw_result)} characters")
        print(f"Processed script length: {len(processed_result)} characters")
        
        return processed_result
        
    except Exception as e:
        print(f"Large model call failed: {str(e)}")
        # If large model call fails, return a sample script for testing
        sample_script = f"[S1]Today we're going to talk about an interesting topic.[S2]Hmm, what topic?[S1]It's about the content we just saw, {content[:100]}...[S2]Oh, sounds very interesting.[S1]Yes, let me give you a detailed introduction."
        print("Using sample script for testing")
        return sample_script


# =============== Main Function ===============

def process_input_to_audio(input_path: str, output_dir: str = "examples"):
    """Complete processing pipeline: from input to audio output
    
    Args:
        input_path (str): Input path (URL, PDF or TXT file)
        output_dir (str): Output directory
    """
    
    # 1. Parse input content
    print("Step 1: Parse input content")
    content = parse_input_content(input_path)
    if not content:
        print("Unable to parse input content, program terminated")
        return
    
    print(f"Content parsed successfully, content preview: {content[:200]}...")
    
    # 2. Use large model to generate dialogue script
    print("\nStep 2: Generate dialogue script")
    script = generate_podcast_script(content)
    if not script:
        print("Dialogue script generation failed, program terminated")
        return

    # 3. Load TTS model
    print("\nStep 3: Load TTS model")
    tokenizer, model, spt = load_model(MODEL_PATH, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device)
    model = model.to(device)
    print("TTS model loading completed")
    
    # 4. Prepare TTS input data
    print("\nStep 4: Prepare TTS input data")
    items = [{
        "text": script,
        "base_path": "",
        "prompt_audio_speaker1": DEFAULT_PROMPT_AUDIO_SPEAKER1,
        "prompt_text_speaker1": DEFAULT_PROMPT_TEXT_SPEAKER1,
        "prompt_audio_speaker2": DEFAULT_PROMPT_AUDIO_SPEAKER2,
        "prompt_text_speaker2": DEFAULT_PROMPT_TEXT_SPEAKER2
    }]
    
    # 5. Set random seed
    # import accelerate
    # accelerate.utils.set_seed(42)
    
    # 6. Generate audio
    print("\nStep 5: Generate audio")
    actual_texts_data, audio_results = process_batch(
        batch_items=items,
        tokenizer=tokenizer,
        model=model,
        spt=spt,
        device=device,
        system_prompt=SYSTEM_PROMPT,
        start_idx=0,
        use_normalize=True
    )
    
    # 7. Save audio files
    print("\nStep 6: Save audio files")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, audio_result in enumerate(audio_results):
        if audio_result is not None:
            output_path = os.path.join(output_dir, f"generated_podcast_{idx}.wav")
            torchaudio.save(output_path, audio_result["audio_data"], audio_result["sample_rate"])
            print(f"Audio saved to: {output_path}")
        else:
            print(f"Audio generation failed: sample {idx}")
    
    print("\nProcessing completed!")


# =============== Usage Examples ===============

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Generate podcast audio: supports URL, PDF or TXT file input")
    parser.add_argument("input_path", help="Input path: URL address, PDF file path or TXT file path")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory (default: outputs)")
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output}")
    
    process_input_to_audio(args.input_path, args.output)