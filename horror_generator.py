# -*- coding: utf-8 -*-
"""
Automated Horror Story and Audiobook Generator

This script generates a long-form psychological horror story using a local LLM,
converts the story into a WAV audio file for an audiobook, and sends progress
updates and the final files via Telegram.

Setup:
1. Install required libraries:
   pip install llama-cpp-python numpy soundfile kokoro-tts requests

2. Set Environment Variables:
   - TELEGRAM_BOT_TOKEN: Your Telegram bot's API token.
   - TELEGRAM_CHAT_ID: The chat ID where notifications and files will be sent.

3. Ensure you have the GGUF model file available. This script is configured to
   download it automatically from the Hugging Face Hub.
"""

import logging
from pathlib import Path
import re
from typing import Dict, List
import gc
from llama_cpp import Llama
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import os
import requests
import json
from datetime import datetime
import threading
import queue
import time

# --- Telegram Configuration ---
# Fetch credentials from environment variables for security
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')

# --- Logging Setup ---
# Basic configuration for console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TelegramLogHandler(logging.Handler):
    """
    Custom logging handler that sends log records to a Telegram chat.
    It uses a queue and a separate worker thread to send messages
    asynchronously, preventing the main application from blocking.
    """
    def __init__(self, bot_token: str, chat_id: str):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.message_queue = queue.Queue()
        self.stop_event = threading.Event()
        # Start a daemon thread to process the log queue
        self.worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self.worker_thread.start()

    def _message_worker(self):
        """Processes the queue and sends messages to Telegram."""
        while not self.stop_event.is_set():
            try:
                message = self.message_queue.get(timeout=1.0)
                # Telegram has a message length limit of 4096 chars
                max_length = 4000
                messages_to_send = [message[i:i + max_length] for i in range(0, len(message), max_length)]

                for msg in messages_to_send:
                    payload = {'chat_id': self.chat_id, 'text': msg, 'parse_mode': 'Markdown'}
                    requests.post(self.url, json=payload, timeout=10)
                    time.sleep(0.1) # Avoid rate limiting
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log errors to the console if Telegram sending fails
                print(f"Error in Telegram worker thread: {e}")

    def emit(self, record):
        """Formats the log record and puts it into the message queue."""
        try:
            log_entry = self.format(record)
            emoji_map = {'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌', 'CRITICAL': '🚨'}
            emoji = emoji_map.get(record.levelname, '🔵')
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Format message for Telegram using Markdown
            telegram_message = f"{emoji} *{record.levelname}* `{timestamp}`\n```\n{log_entry}\n```"
            self.message_queue.put(telegram_message)
        except Exception as e:
            print(f"Error formatting log for Telegram: {e}")

    def close(self):
        """Stops the worker thread gracefully."""
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        super().close()


def setup_telegram_logging(bot_token: str, chat_id: str) -> TelegramLogHandler | None:
    """Verifies Telegram connection and sets up the custom log handler."""
    try:
        # Check if the bot token is valid by calling the getMe endpoint
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
        if not response.ok:
            print(f"Failed to connect to Telegram bot API. Status: {response.status_code}")
            return None
        
        bot_info = response.json()
        print(f"Successfully connected to Telegram bot: {bot_info.get('result', {}).get('username', 'Unknown')}")
        
        telegram_handler = TelegramLogHandler(bot_token, chat_id)
        telegram_handler.setLevel(logging.INFO)
        # A simple formatter is best, as the handler adds its own rich formatting
        telegram_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
        return telegram_handler
    except Exception as e:
        print(f"An error occurred while setting up Telegram logging: {e}")
        return None


def send_telegram_message(bot_token: str, chat_id: str, message: str):
    """A general-purpose function to send a Markdown-formatted message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


def send_audio_file_to_telegram(bot_token: str, chat_id: str, audio_file_path: str, title: str):
    """
    Sends an audio file to the specified Telegram chat, with progress updates
    and handling for file size limits.
    """
    try:
        logger.info(f"Preparing to send audio file to Telegram: {audio_file_path}")
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False

        file_size = os.path.getsize(audio_file_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")

        # Telegram Bot API has a 50MB limit for files sent by bots
        if file_size > 50 * 1024 * 1024:
            logger.warning(f"File too large for Telegram ({file_size_mb:.2f} MB). Maximum is 50MB.")
            message = (f"🎵 *Audio File Ready*\n\n"
                       f"*Title:* {title}\n"
                       f"*Size:* {file_size_mb:.2f} MB\n"
                       f"*Status:* ❌ File too large for Telegram\n"
                       f"*Location:* `{audio_file_path}`\n"
                       f"*Time:* {datetime.now().strftime('%H:%M:%S')}")
            send_telegram_message(bot_token, chat_id, message)
            return False

        start_message = (f"🎵 *Uploading Audio File*\n\n"
                         f"*Title:* {title}\n"
                         f"*Size:* {file_size_mb:.2f} MB\n"
                         f"*Status:* Uploading...\n"
                         f"*Time:* {datetime.now().strftime('%H:%M:%S')}")
        send_telegram_message(bot_token, chat_id, start_message)

        url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': (os.path.basename(audio_file_path), audio_file, 'audio/wav')}
            caption = (f"🎵 *{title}* - Horror Story Audiobook\n\n"
                       f"📊 File size: {file_size_mb:.2f} MB\n"
                       f"⏱️ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            data = {'chat_id': chat_id, 'title': title, 'caption': caption, 'parse_mode': 'Markdown'}
            
            # Set a generous timeout based on file size to prevent premature failures
            timeout = max(300, int(file_size_mb * 15))
            
            response = requests.post(url, files=files, data=data, timeout=timeout)
            
            if response.ok:
                logger.info("✅ Audio file successfully sent to Telegram!")
                return True
            else:
                logger.error(f"Failed to send audio file: {response.status_code} - {response.text}")
                error_message = (f"❌ *Audio Upload Failed*\n\n"
                                 f"*Title:* {title}\n"
                                 f"*Error:* {response.status_code} - Upload failed\n"
                                 f"*File Location:* `{audio_file_path}`\n"
                                 f"*Time:* {datetime.now().strftime('%H:%M:%S')}")
                send_telegram_message(bot_token, chat_id, error_message)
                return False

    except requests.exceptions.Timeout:
        logger.error("Timeout while uploading audio file to Telegram.")
        timeout_message = (f"⚠️ *Audio Upload Timeout*\n\n"
                           f"*Title:* {title}\n"
                           f"*Status:* Upload timed out\n"
                           f"*File Location:* `{audio_file_path}`\n"
                           f"*Time:* {datetime.now().strftime('%H:%M:%S')}")
        send_telegram_message(bot_token, chat_id, timeout_message)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred sending audio to Telegram: {e}")
        return False


def extract_story_idea_from_prompt(prompt_text: str) -> Dict[str, str]:
    """Parses a structured prompt text into a dictionary."""
    idea = {'title': 'Untitled Horror Story'}
    for line in prompt_text.strip().split('\n'):
        if line.startswith('# '):
            idea['title'] = line[2:].strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            # Normalize keys to be used as dictionary keys
            key = key.strip().lower().replace(' ', '_').replace('-', '_')
            idea[key] = value.strip()
    return idea


class SuburbanHorrorGenerator:
    """
    A class dedicated to generating a structured, multi-act horror story
    by iteratively calling an LLM and maintaining context.
    """
    def __init__(self, llm_instance: Llama):
        self.llm = llm_instance
        self.section_summaries = []
        self.current_story_data = {}
        self.previous_section_text = ""
        self.health_threshold = 0.7 # Min percentage of target words to be considered valid
        self.max_retries = 3

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        """Defines the 6-act structure of the story based on the initial idea."""
        protagonist = story_idea.get('protagonist', 'a curious teenager')
        antagonist = story_idea.get('antagonist', 'seemingly normal neighbors')
        setting = story_idea.get('setting', 'a quiet suburban neighborhood')
        victim = story_idea.get('victim', 'the girl next door')
        secret = story_idea.get('secret', 'a horrifying truth hidden in plain sight')

        # A detailed 6-act structure ensures a well-paced and developed plot
        acts = [
            {"act": 1, "title": "The Ordinary World", "sections": [
                {"section": 1, "title": "The Neighborhood", "description": f"Introduce {protagonist} and the seemingly perfect {setting}. Establish the peaceful facade and hint at underlying tension.", "target_words": 600},
                {"section": 2, "title": "First Encounters", "description": f"{protagonist} meets the strange {victim} and the overly friendly {antagonist}. Their interactions feel normal, but with an unsettling edge.", "target_words": 600}
            ]},
            {"act": 2, "title": "Inciting Incidents", "sections": [
                {"section": 1, "title": "Strange Occurrences", "description": f"{protagonist} witnesses odd behaviors and inconsistencies from the {antagonist}, raising the first real suspicions.", "target_words": 600},
                {"section": 2, "title": "Dismissal and Isolation", "description": f"When {protagonist} voices concerns, adults dismiss them as imagination. {protagonist} starts to feel isolated as {victim} shows more signs of distress.", "target_words": 600}
            ]},
            {"act": 3, "title": "The Point of No Return", "sections": [
                {"section": 1, "title": "The First Glimpse of Horror", "description": f"{protagonist} accidentally witnesses an undeniable act that confirms their fears about what is happening to {victim}.", "target_words": 600},
                {"section": 2, "title": "Moral Paralysis", "description": f"Reeling from the discovery, {protagonist} is paralyzed by fear and the weight of the secret, struggling with what to do next.", "target_words": 600}
            ]},
            {"act": 4, "title": "Descent into Darkness", "sections": [
                {"section": 1, "title": "Uncovering the Awful Truth", "description": f"Driven by the need to know more, {protagonist} investigates and uncovers the full, systematic nature of the {secret}.", "target_words": 600},
                {"section": 2, "title": "The Horrifying Complicity", "description": f"{protagonist} makes the shocking discovery that other neighbors know about the {secret} and are complicit in the silence.", "target_words": 600}
            ]},
            {"act": 5, "title": "The Climax", "sections": [
                {"section": 1, "title": "A Desperate Plan", "description": f"Knowing no one will help, {protagonist} devises a risky plan to confront the {antagonist} and save {victim}.", "target_words": 600},
                {"section": 2, "title": "The Confrontation", "description": f"The plan is set in motion, leading to a direct and violent confrontation where the suburban facade is shattered completely.", "target_words": 600}
            ]},
            {"act": 6, "title": "The Aftermath", "sections": [
                {"section": 1, "title": "The Reckoning", "description": f"The immediate aftermath of the confrontation. The authorities finally get involved, and the truth comes to light for the whole town.", "target_words": 600},
                {"section": 2, "title": "A Haunted Memory", "description": f"The story concludes by exploring the lasting psychological scars on {protagonist}, {victim}, and the community, which is forever changed.", "target_words": 600}
            ]}
        ]
        return acts

    def _build_comprehensive_context(self) -> str:
        """Constructs a detailed context prompt for the LLM to ensure story consistency."""
        # Core character and plot elements must remain consistent
        char_context = f"""**CHARACTER AND PLOT BIBLE (MAINTAIN CONSISTENCY):**
- **Protagonist:** {self.current_story_data.get('protagonist')}
- **Antagonist:** {self.current_story_data.get('antagonist')}
- **Victim:** {self.current_story_data.get('victim')}
- **Setting:** {self.current_story_data.get('setting')}
- **The Central Secret:** {self.current_story_data.get('secret')}"""

        # A summary of past events helps the model track the plot
        plot_context = ""
        if self.section_summaries:
            plot_context = "\n\n**STORY SO FAR (SUMMARY):**\n" + "\n".join(self.section_summaries)

        # The last few sentences provide a direct lead-in for a seamless transition
        recent_context = ""
        if self.previous_section_text:
            words = self.previous_section_text.split()
            # Use last 150 words for immediate context
            recent_words = words[-150:]
            recent_context = f"\n\n**CONTINUE DIRECTLY FROM THIS TEXT:**\n...{' '.join(recent_words)}"

        return f"{char_context}{plot_context}{recent_context}"

    def _clean_text(self, text: str) -> str:
        """Removes unwanted artifacts, prompt remnants, and cleans up the LLM's output."""
        # This is a common fix for encoding errors from some models
        try:
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        text = text.strip()
        
        # Remove any leftover instructional phrases from the model's output
        unwanted_patterns = [
            r'^\s*act \d+, section \d+.*$',
            r'^\s*\[end of section\].*$',
            r'^\s*continue with act.*$',
            r'^\s*WRITING STYLE:.*$',
            r'^\s*CRITICAL INSTRUCTIONS:.*$'
        ]
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        return text.strip()

    def _validate_section(self, section_text: str, target_words: int) -> bool:
        """Checks if the generated text is of sufficient length and quality."""
        if not section_text or section_text.lower().startswith("[error"):
            return False

        word_count = len(section_text.split())
        min_words = int(target_words * self.health_threshold)

        if word_count < min_words:
            logger.warning(f"Validation failed: Word count {word_count} is below the minimum of {min_words}.")
            return False

        return True

    def generate_section(self, act_num: int, section_num: int, section_data: Dict[str, str], retry_count: int = 0) -> str:
        """Generates a single section of the story with a retry mechanism."""
        title = section_data["title"]
        description = section_data["description"]
        target_words = section_data["target_words"]

        # A detailed style guide helps the model maintain the correct tone
        horror_style_guide = (
            "**WRITING STYLE GUIDE: Suburban Psychological Horror**\n"
            "- Focus on the horror lurking beneath mundane, everyday normalcy.\n"
            "- Build suspense and dread through psychological tension, not gore.\n"
            "- Emphasize the quiet complicity and moral cowardice of 'normal' people.\n"
            "- Use realistic dialogue and character motivations.\n"
            "- Explore themes of lost innocence and the darkness hidden in plain sight."
        )

        context = self._build_comprehensive_context()

        prompt = f"""You are a master horror writer. Continue the following psychological horror story seamlessly.
Maintain absolute consistency with all established characters, plot points, and tone. Do not repeat information.

**STORY TITLE:** "{self.current_story_data['title']}"

{context}

---
**CURRENT TASK:**
Write the next section of the story: **Act {act_num}, Section {section_num} - {title}**
**SECTION GOAL:** {description}
**TARGET WORD COUNT:** Approximately {target_words} words.

{horror_style_guide}

**INSTRUCTIONS:**
- Write only the story text. Do not include titles, headers, or any meta-commentary.
- Begin writing the story continuation immediately.

**STORY CONTINUATION:**
"""

        logger.info(f"Generating Act {act_num}, Section {section_num}: '{title}' (Attempt {retry_count + 1}/{self.max_retries})")

        try:
            output = self.llm(
                prompt,
                max_tokens=int(target_words * 2.5),  # Allow ample space for generation
                temperature=0.55,
                top_p=0.9,
                repeat_penalty=1.12,
                stop=["\n\n\n", "---", "##", "THE END", "***"], # Stop tokens to prevent rambling
                echo=False
            )

            section_text = self._clean_text(output['choices'][0]['text'])

            if self._validate_section(section_text, target_words):
                word_count = len(section_text.split())
                logger.info(f"  ✓ Act {act_num}, Section {section_num} completed with {word_count} words.")
                self.previous_section_text = section_text
                summary = f"Act {act_num}-{section_num} ({title}): {description[:80]}..."
                self.section_summaries.append(summary)
                return section_text
            else:
                logger.warning(f"  ✗ Validation failed for Act {act_num}, Section {section_num}.")
                if retry_count < self.max_retries - 1:
                    logger.info("Retrying generation...")
                    time.sleep(2) # Brief pause before retrying
                    return self.generate_section(act_num, section_num, section_data, retry_count + 1)
                else:
                    error_msg = f"[Error: Failed to generate valid content for Act {act_num}, Section {section_num} after {self.max_retries} attempts.]"
                    logger.error(error_msg)
                    return error_msg

        except Exception as e:
            logger.error(f"  ✗ An exception occurred during generation for Act {act_num}, Section {section_num}: {e}", exc_info=True)
            return f"[Error: Generation failed due to an exception.]"

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        """Orchestrates the generation of the entire story from start to finish."""
        logger.info(f"\n=== STARTING STORY GENERATION: '{story_idea['title']}' ===")
        
        # Reset state for a new story
        self.current_story_data = story_idea
        self.section_summaries = []
        self.previous_section_text = ""

        if telegram_handler:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                f"📖 *Starting Story Generation*\n\n*Title:* {story_idea['title']}\n*Structure:* 6 acts, 12 sections"
            )

        story_structure = self.create_story_structure(story_idea)
        full_story_sections = []
        total_words = 0

        for act in story_structure:
            act_num = act['act']
            act_title = act['title']
            logger.info(f"\n--- Starting Act {act_num}: {act_title} ---")
            
            # Add a separator between acts for readability
            if act_num > 1:
                full_story_sections.append("\n\n* * *\n\n")

            for section in act['sections']:
                section_num = section['section']
                section_text = self.generate_section(act_num, section_num, section)

                if "[Error:" not in section_text:
                    full_story_sections.append(f"{section_text}\n\n")
                    total_words += len(section_text.split())
                else:
                    # Append the error message to the story to know where it failed
                    full_story_sections.append(f"_{section_text}_\n\n")
                
                # Explicitly collect garbage to free up memory, crucial for long runs
                gc.collect()

        complete_story_text = "".join(full_story_sections)

        logger.info("\n=== STORY GENERATION COMPLETE ===")
        logger.info(f"Total words generated: {total_words}")

        if telegram_handler:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                f"✅ *Story Generation Completed!*\n\n*Title:* {story_idea['title']}\n*Total Words:* {total_words}"
            )

        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words
        }


def clean_for_audio(text: str) -> str:
    """Prepares the final story text for the text-to-speech engine."""
    logger.info("Cleaning story text for audio narration...")
    # Remove markdown like headers, bold, italics, and separators
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'^\s*(\* \* \*|---)\s*$', '', text, flags=re.MULTILINE)
    # Normalize whitespace to prevent long pauses
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    logger.info("Text cleaning for audio complete.")
    return text


def generate_audio(text: str, title: str, output_file: str) -> str | None:
    """Generates a WAV audio file from text using the Kokoro TTS engine."""
    try:
        logger.info("Initializing audio generation...")
        if telegram_handler:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                f"🎤 *Starting Audio Generation*\n\n*Title:* {title}\n*Status:* Initializing TTS engine..."
            )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 'a' lang_code in kokoro means automatic language detection
        pipeline = KPipeline(lang_code='a')
        audio_segments = []

        # Split text into chunks (e.g., by paragraph) for stable processing
        text_chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]
        total_chunks = len(text_chunks)
        logger.info(f"Processing {total_chunks} text chunks for audio generation.")

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing audio chunk {i + 1}/{total_chunks}")
            
            # Send progress update to Telegram every 10 chunks
            if telegram_handler and (i + 1) % 10 == 0:
                percentage = ((i + 1) / total_chunks) * 100
                progress_message = (f"🎤 *Audio Generation Progress*\n\n"
                                    f"*Title:* {title}\n"
                                    f"*Status:* Processing chunk {i + 1}/{total_chunks} ({percentage:.1f}%)")
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, progress_message)

            # Generate audio for the chunk
            generator = pipeline(chunk, voice='bm_fable') # 'bm_fable' is a good narrative voice
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)

        if audio_segments:
            # Concatenate all audio segments and write to a WAV file
            combined_audio = np.concatenate(audio_segments, axis=0)
            sample_rate = 24000  # Kokoro's default sample rate
            sf.write(output_file, combined_audio, sample_rate)
            duration_sec = len(combined_audio) / sample_rate
            logger.info(f"Audio saved to '{output_file}' - Duration: {duration_sec:.2f} seconds")

            if telegram_handler:
                send_telegram_message(
                    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                    f"✅ *Audio Generation Completed!*\n\n*Title:* {title}\n*Duration:* {duration_sec:.2f}s\n*File:* `{output_file}`"
                )
            return output_file
        else:
            logger.warning("No audio segments were generated. The text might have been empty.")
            return None

    except Exception as e:
        logger.error(f"A critical error occurred during audio generation: {e}", exc_info=True)
        if telegram_handler:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                f"❌ *Audio Generation Failed*\n\n*Title:* {title}\n*Error:* An unexpected error occurred."
            )
        return None

# --- Main Execution ---
def main():
    """Main function to run the entire story generation and audiobook creation pipeline."""
    global telegram_handler
    
    # Setup Telegram logging if tokens are configured
    if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
        telegram_handler = setup_telegram_logging(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if telegram_handler:
            logger.addHandler(telegram_handler)
            logger.info("🚀 Horror story generator starting! Telegram logging is active.")
        else:
            logger.warning("Failed to initialize Telegram logging. Continuing with console only.")
    else:
        logger.info("Telegram tokens not configured. Logging to console only.")

    try:
        # --- 1. Load the Language Model ---
        logger.info("Loading Llama model... This may take a moment.")
        # Using from_pretrained will automatically download the model if not found locally
        llm = Llama.from_pretrained(
            repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
            filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
            n_ctx=8192,      # 8k context window is good for long stories
            n_batch=256,     # Batch size for prompt processing
            n_threads=None,  # None lets llama.cpp decide the optimal number of threads
            n_gpu_layers=-1, # Offload all possible layers to GPU
            verbose=False    # Set to True for detailed llama.cpp logs
        )
        logger.info("✅ Model loaded successfully.")

        # --- 2. Define the Story Prompt ---
        story_prompt = """
        # The House on Maple Street
        Protagonist: David Parker, a 16-year-old boy who recently moved to Millbrook with his family. He's observant, curious, and struggles to fit in.
        Antagonist: Ruth and Richard Chandler, a middle-aged couple who appear to be model citizens. They live in the immaculate house next door.
        Setting: Millbrook, a picturesque suburban town where everyone knows everyone, and dark secrets hide behind white picket fences.
        Victim: Meg Loughlin, a 14-year-old girl who lives with the Chandlers as their "niece." She's quiet, withdrawn, and always seems afraid.
        Secret: The Chandlers are systematically torturing and abusing Meg, while the entire neighborhood either ignores the signs or actively covers it up to maintain their perfect community image.
        """
        
        # --- 3. Generate the Story ---
        story_idea = extract_story_idea_from_prompt(story_prompt)
        generator = SuburbanHorrorGenerator(llm)
        final_story = generator.generate_complete_story(story_idea)

        if not final_story or final_story['word_count'] == 0:
            logger.error("Story generation failed to produce content. Exiting.")
            return

        # --- 4. Save Story and Prepare for Audio ---
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        story_title_slug = final_story['title'].replace(' ', '_').replace("'", "")

        # Save the full story as a Markdown file
        story_file_md = output_dir / f"{story_title_slug}.md"
        with open(story_file_md, 'w', encoding='utf-8') as f:
            f.write(f"# {final_story['title']}\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Word Count:** {final_story['word_count']}\n\n---\n\n{final_story['text']}")
        logger.info(f"Full story saved to: {story_file_md}")

        # --- 5. Generate the Audiobook ---
        audio_text = clean_for_audio(final_story['text'])
        audio_file_path = str(output_dir / f"{story_title_slug}_Audiobook.wav")
        
        audio_file = generate_audio(audio_text, final_story['title'], audio_file_path)

        # --- 6. Send Audiobook to Telegram ---
        if audio_file and telegram_handler:
            logger.info(f"Audio file generation successful. Proceeding to send to Telegram.")
            send_audio_file_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, audio_file, final_story['title'])

        if telegram_handler:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"🎉 *All tasks completed for '{final_story['title']}'!*")

    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution loop: {e}", exc_info=True)
        if telegram_handler:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"🚨 *CRITICAL ERROR*\n\nThe script has crashed. Please check the logs.\n`{e}`")
        raise

if __name__ == "__main__":
    try:
        main()
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.error(f"Script exited with a critical failure.")
        # The logger already captured the detailed exception, so we can exit.
        exit(1)
    finally:
        # Ensure the Telegram logger is closed gracefully
        if 'telegram_handler' in globals() and telegram_handler is not None:
            telegram_handler.close()

