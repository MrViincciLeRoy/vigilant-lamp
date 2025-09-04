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
audio_text = "Chapter 1: Born of the Comet

The rain lashed through the forest at midnight, blurring the edges of acacia and baobab under a storm-heavy sky. Kagiso, fourteen and lean, ran on instinct through the thigh-high ferns, branches slashing at his face and arms. The earth underfoot was slick mud, each step threatening to slide him into a tangle of roots. Lightning split the sky overhead, turning the forest momentarily into a jagged silhouette of gray and black. Thunder boomed behind him, rattling the bones in his chest. Kagiso didn’t notice. He was focused on the pounding of his own heart.

Behind him, shapes crept in the darkness: Shadow Children, unnatural ghosts born of the Dark Circle. Pale-flickering figures floated between the tree trunks, their glowing eyes burning through the black. A chorus of whispered voices wound around Kagiso’s mind: “No one hears you…” “Trapped forever…” Another cried with a child’s giggle that twisted into a hyena’s bark. A cold dread slithered down his spine. He could feel the shadows closing in, hungrily feeding on his fear.

Kagiso’s legs pumped faster. He risked a glance back: the wraithlike children flitted through the rain, stalking him. Slender limbs—colored a sickly grey—reached out, jagged fingers clawing at the air. One of them shrieked, a sound like tearing cloth, and Kagiso felt the air around him thicken. He knew their name for their power: Whispered Fear. The voices wormed under his skin, but he refused to slow. I didn’t survive this long to freeze now, Kagiso thought, pressing himself onward.

He was alone. Always alone, since the day they threw him from the village gates like garbage. Five long years in this storm-swamped wilderness, sleeping under twisted roots and foraging for baobab fruit, eating bugs to survive. Nights were cold and empty; all he had was the sky and its vastness to keep him company. When hunger clawed at him in the night, he remembered the naledi (star) that burned across the sky the night he was born. The priests had wrapped him in goat-skin at the village altar under a blood-red comet. They called him Mwana wa Naledi ya Phirima—Child of the Fire Star. He could almost hear his mother’s lullaby, softly sung in Setswana to the Badimo (ancestors) by firelight, until dawn took the song from her lips.

Kagiso’s throat tightened. His lungs burned with each ragged breath. The forest around him felt alive, watching. Every sharp crack of twig, every distant owl cry told him where the shadows lurked. He weaved between roots and fallen logs by memory, each turn calculated. He remembered his old hiding places, blood pounding in his ears. Water pounded in his boots as he rounded a bend—up ahead, he recognized it: the dry bed of a creek, stones polished by past floods. If he could stay in the water, the Shadow Children might lose his scent.

He veered to the left and splashed into the creek. Rainwater gushed around his ankles as he plunged into the shallow stream. Cold stabbed through his soaked skin, but it was a welcome jolt. The water muffled the rustling of leaves, drowning out whispers of doubt. Kagiso ran on, the current churning around his calves, lightning briefly illuminating the riverbed with each flash.

He dared a quick glance back. One of the Shadow Children lunged to catch up. Its eyes glowed like ember coals, and it slipped on a mossy rock with a wet squish. Kagiso seized the chance. He grabbed a fist-sized rock from the forest floor and hurled it backward with all his might.

The rock hurtled through the air and struck the creature’s ghostly arm with a crack like thunder. The creature shrieked—an angry, piercing shriek—and stumbled back. Kagiso followed up; with a swift kick, he sent the wraith tumbling into the shallow water. It gave one final gurgling wail and dissolved into ragged clouds of black mist.

A triumphant grin crossed Kagiso’s face. “Outsmarted you,” he whispered over the thunder. He scrambled over a fallen termite log that bridged the creek, hands steadying him on the wet, grainy wood. Branches snapped underfoot as he sprinted. Behind him, the Shadow Children hissed and flitted like angry moths around the bright trace of his passing. They were relentless; but he was no easy prey.

He didn’t think of slowing. The forest suddenly opened into the mouth of a narrow ravine. It gaped ahead of him like the throat of some giant, one side a sheer wall of limestone, the other jagged with fallen stones and vines. The ravine was black and endless. Kagiso’s chest heaved. There was nowhere left to run. The trees overhead wove into a dense roof of thorns, blocking even the starlight. He skidded to a stop at the ravine’s edge, mud flying around his wet boots, back pressed against the cold rock.

Every instinct screamed to fight. Kagiso gripped his spear tight. The rain hammered the stone around him. Silence pressed in; even the storm’s roar had dimmed under the stone canopy. “I didn’t survive this long to die here,” he told himself, though the words were lost to the thunder.

A soft, dissonant giggle filled the air—a child’s taunt in the cramped, dark space. Kagiso felt his heart skip a beat. His eyes darted upward. Lightning split the sky again, and in its brief flare he saw them: three Shadow Children perched on the ravine walls above, pale fingers probing the muddy water below, limbs thinner than birds’ yet holding them. Their eyes were wide and shining with hunger. The tallest shook its head and tilted it, and all three whispered in brittle chorus: “Always alone…”

Horror knotted in Kagiso’s gut. One of the creatures raised a bony hand and pointed at him. No escape… the whisper slithered in his mind, ice crawling over his blood. Kagiso snapped his teeth shut on the fear that threatened to choke him. He refused to give in.

“Ma…” A name shattered the chant. Kagiso’s mind was swept back to another time—a tiny boy in a grand hut, lullabies and smoke, his mother’s voice in the dark. He blinked and it was gone. He was in the ravine again, clutching his spear. His voice was gone, swallowed by the storm.

Lightning flared across the sky, a wall of white light in the darkness. Kagiso’s eyes narrowed as something roared inside him: a spark of the old power in his blood. He had no time to think or hesitate. With a cry that tore from his soul, Kagiso screamed into the roaring storm.

Then silence. Kagiso remained crouched, spear in hand, senses straining into the darkness.

The old Sangoma—her weathered face as still as a panther’s—appeared at his side. She was silent at first, waiting. Then a voice like a drumbeat rolled out of the night: “Kagiso… the Badimo heard you.” In the trembling dark, Kagiso only stared.

“Come,” the Sangoma whispered, guiding him back down the ravine. Kagiso followed, stumbling, each breath a wet rasp in his lungs. The fireflies were gone; only rainwater and thunder were left.

Kagiso pressed himself into the deepest shadow of the ravine wall, his damp body against the cool limestone. The rain drummed above him, but here the sound was muted by rock. Each breath he took was sharp and ragged. He could taste the metal of blood on his tongue. Fingers white-knuckled around the haft of his spear, he tried to steady himself. Across from him, the jagged wall climbed out of sight, draped with moss and tangled vines. The place felt empty and ancient, like a stone tomb.

Above him, on narrow rocky ledges, the three Shadow Children had regrouped. They peered down at him with glowing eyes and wide, hungry grins. Lightning flashed again. Kagiso saw their cruel faces—sunken cheeks and wicked smiles—and felt another stab of dread in his gut. One of them hissed a soundless laugh that echoed in the darkness, as cold and empty as the grave. Then, with fluid motion, one of the shadows dropped into the ravine with silent wings unfurling to catch its fall.

“Ga go na kganyogo,” Kagiso muttered under his breath in Setswana (No mercy). He planted his bare feet in the mud and drew himself up to full height. He was surrounded. Each thump of water on stone was the heartbeat of the oncoming fight. His spear felt heavy and solid in his hands—a reminder that this was real, and that tonight he would make his stand.

The first Shadow Child leapt down into the muddy bank directly in front of him, claws outstretched. Kagiso reacted without thought. He braced with one foot and drove the tip of his spear forward as if spearing air—but in that instant, he drove it true. The shaft slid through the vaporous form with a chilling ting. The creature let out a sharp, keening shriek as if stabbed through flesh of glass. Kagiso felt nothing but cold air in his hands, but he followed through. With a grunt he kicked the phantom in the chest, sending it stumbling backward into the water. It slumped and with one final hiss of despair it erupted in a dark plume and vanished. The shriek faded into silence.

Two remained. One of them lunged at his side, talons slicing through the air, while the other crept up behind, nails clicking on stone. Kagiso twisted away from the first, but the second had him in its sight now. He spun in one fluid motion, facing the second attacker as he turned back to meet the other. The two shadows circled, waiting like jackals.

Kagiso’s memories sharpened for an instant: days of training with a sliver of wood in the village yard, his aunt showing him how to hold a spear, how to strike a target. He snapped back to the present. He feinted left and then right, swinging his spearhead upward like a battle hammer. It connected with a cracking swipe. The ghost thrashed, its mouth opening in a wordless scream as dark ichor spilled onto the ground beneath it. It slumped forward, and in a final gurgling breath it disappeared into ash and mist.

For a breathless moment, Kagiso stood alone among the shafts of rain and shattered leaves. Only the final Shadow Child remained. It skittered back along the ravine wall, and looked down at him, its face twisted in surprise and rage. Kagiso’s chest swelled as the last echoes of panic drained out of him, replaced by sharp focus. This ends now, he vowed.

Lightning split the sky one more time. In its brief flash, the last phantom reared back and darted toward a low outcropping above. It lashed out with brittle teeth, and Kagiso charged. He felt raw power surging through his veins. With a roar, he leapt forward, spear raised high. The tip jabbed toward nothingness—the Shadow Child was gone. In its place came only a shrill teeee! as its spirit dissolved in the storm’s fury, the echo swallowed by the rain.

Kagiso staggered back to his knees. The silence returned like a wave crashing to shore. He was alone; the Shadow Children were gone.

Only then did the adrenaline drain away. Kagiso’s chest heaved with exhaustion. He spat gritty blood into the mud. His vision swam with spots of gray. As the spinning slowed, he realized the deafening silence was broken only by the soft drip of his own blood into the puddle below.

He tried to stand, and sagged. His arm throbbed where he must have been scratched—he tugged at the sleeve. A ribbon of crimson ran down his forearm from a deep gash. His fingers pressed into it; the pain made him wince, but he breathed evenly. He looked down and saw more: a series of slash marks across his calf, dark maroon against faded denim. Each wound sent electricity thundering through him. He was bleeding, badly.

The rain pelted his wounds. The cold seeped into his bones, starting a slow ache. Muscles trembled from the fight and the fall. Kagiso pressed his palm to the wall, steadying himself. He had won this battle... and yet exhaustion drew him down like a heavy cloak.

Lightning flashed again. He blinked up at a shredded vine dangling above, the distant moon behind drifting clouds. One faint fear remained: he was utterly alone here, with no one to help, no place to go. He would have to walk out of the ravine on his own, wounded as he was. The thought would have broken a weaker boy. Kagiso growled to himself in stubborn defiance: “Ga ba tseye pelo, ga ke kitla ka bolawa fa.” (They won’t break my heart; I will not die here.)

He sank fully to the ground, drawn in by pain. The spear slipped from his numb fingers and clattered on the stones. He wrapped both arms around his knees, trying to still the trembling. He matched his heartbeat to the rhythm of dripping rain, fighting to steady his nerves.

Kwana, ngwana wa naledi...
A voice softly spoke in his head, soothing yet urgent. Kagiso’s eyes snapped open. In the cold dim of the ravine, he recognized the tone of his mother. “Kwana, ngwana wa naledi,” it repeated—(Stop, child of the star). The darkness in his head began to hum with a gentle power.

Then another voice, patient and ancient, spoke plainly as if right beside him: “Call us, child of the comet.” The air around him grew warm. His cuts stung as if kissed by fire, and blue-white light flared along the edges of his wounds.

Kagiso exhaled slowly; something ancient had stirred in the darkness.

Kagiso’s scream tore through the ravine like the crack of thunder itself. For a heartbeat, time seemed to bend—the raindrops froze midair, and the forest around him fell silent, watching. A jagged scar of light lanced down from the storm clouds above, converging on him. The anger and fear that had built up all those lonely years surged out with that cry, and the comet’s fire ignited in his veins.

Suddenly, a roaring heat bloomed in his chest. Kagiso felt blood and water ripple beneath his feet as if the earth were bowing. His eyes snapped shut under the brilliance. When he opened them, the world shone in cobalt and white. He could feel nothing but raw power: it coursed through his arms, legs, every heartbeat a thunderous drum. He gasped, blood warming in his mouth, but he couldn’t remember ever feeling anything so alive—even when he had first tasted freedom under a different sky.

His head buzzed. Sweat and rain mixed on his brow as lightning danced across the stone wall of the ravine, painting Kagiso’s face in flickers of blue-white light. He lifted his head. There was movement beside him, a form emerging from the storm’s glow.

The spirit stood taller than any man, its form wreathed in stars and smoke. It looked like a warrior king from the old stories: broad shoulders, strong jaw, an expression of fierce nobility. In its hand it held a spear ignited in primordial flame, long and sharp as a titan’s tooth. Kagiso recognized the face pattern in its eyes: a Badimo ancestor come out of legend—perhaps a long-lost chief or hero of his clan, summoned to do battle once more.

Kagiso barely had time to meet its gaze. A hand seemed to rest on his shoulder, not to comfort, but to let him know the power was his to command. With a single, wordless nod from the spectral warrior, Kagiso felt both fear and exaltation surge through him. He gave a fierce grin and rose to his full height despite the pain, tasting iron and smoke on his tongue.

All around, the forest responded. The vines that climbed the ravine walls twisted upward as if reaching to the heavens. The rafters of the sky echoed Kagiso’s heartbeat. He planted his spear tip into the muddy ground and raised his free hand to the air.

The specter-ancestor hurled a spear of pure light, and Kagiso roared in triumph. A blinding bolt of energy shot into the brooding storm cloud above. It seemed to pierce the very heart of darkness. Thunder exploded overhead, closer and louder than ever, and rain came down hard in a sudden drench.

Kagiso stomped the ground. A shockwave rippled outward. Water and earth flew like spray. Birds shrieked and scattered as the once-dark ravine turned suddenly bright. He felt the electricity of the storm tasting sweet on his tongue.

From the rim of the ravine, weak wails answered—terrified cries of the Shadow Children. They clung to the edges now, shrinking away from the intolerable light.

One of them—a gaunt creature with a twisted grin—darted out of hiding, as if daring Kagiso to try again. In an instant, the specter’s flaming spear traced a great arc in the air. The blow was swifter than a striking serpent: it cleaved the shadow child in two. A hiss and pop like burning coal, and in a thunderous plume it vanished into nothing.

Another leaped at Kagiso’s roaring flame, but the warrior-spirit twirled and threw the spear like a javelin across the ravine. The spear winked through the night—a streak of falling star—and impaled the spirit-creature in the chest. With a final screech it disintegrated in a shower of sparks.

Rising thunder blotted out any other sound. Only Kagiso’s ragged breathing remained. He realized it as the last of the Shadow Children collapsed and evaporated. The ravine was utterly still.

Kagiso lowered his spear slowly. Lightning flashed again, and for a moment he saw everything drenched in sapphire luminescence. Cracks spider-webbed through the stone walls from where spears and electricity had struck. The edges of his vision shimmered with starfire. He looked down at himself. Every old knife cut and bruise he had carried was gone. His skin, once bruised and bleeding, glowed faintly gold in the lightning. The pain in his shoulder and leg had vanished, replaced by a warm tingling. He flexed his fingers and found them steady. Even his clothes were dry, though the rain was pouring.

For the first time in his life, Kagiso felt neither fear nor despair. He felt whole—alive, free, at peace. He imagined he could feel the gaze of every ancestor who had ever lived in his blood, proud and protective.

His knees suddenly gave way. He fell into the wet dirt, one hand catching himself. It was all he could do to stay upright.

Lightning crackled violently above, and in that final flash Kagiso’s whole being was radiant. A lone crow perched on a vine shrieked its caw into the storm. For a split second, Kagiso’s body shone like a living torch, and in his vision he was a warrior stepping through the gate of stars.

Then Kagiso dropped to his knees. He closed his eyes, trying to still the power, to let the calm of the night cover him. The storm’s thunder had moved on as quickly as it came. In its wake, the first hints of dawn’s light crept into the sky above the ravine. Only the gentle patter of the finishing rain remained.

He spat into the mud. Blood. The ghost fires had cut him as they had the shadows. A thin trickle slid down his chin. Kagiso wiped it away. His voice returned in a raspy whisper.

“I… I lived,” he said to the wind, three words that felt like the weight of a lifetime. The ancestors had heard him. The storms had answered.

His body vibrated with exhaustion, but he put a hand to his chest and felt only his pounding heart and that faint warmth of victory. The spear beside him felt heavy and sacred all at once. Lightning-split silhouettes of broken trees and a shattered vine ladder framed him.

He drew in a shaky breath. It would be dawn soon; the summer moon was a pale sliver fading to gray. There was nothing left to fight in these woods tonight.

Kagiso stood slowly to his feet. Every muscle screamed in protest, but his back went ramrod straight. He dusted the dirt from his pants. Rain had stopped falling; droplets hung from his lashes.

Somewhere far off, a pack of hyenas yelped, as if celebrating. Kagiso clutched his spear tighter. No one would ever know what happened here tonight. The forest would forget it. But he would never forget.

He stepped out of the ravine into the tall grass and headed toward the distant glimmer of the village. But the world swam before his eyes, and darkness claimed him before he could reach safety.

His first conscious sensation was cool water dripping onto his brow. When Kagiso opened his eyes again, he saw the thatched roof of a mud-brick hut above him. The air smelled of herbs and smoke. He felt tangled in a woven mat. His head throbbed, and he was dizzy with hunger and pain. A woman he did not know knelt beside him, stirring a clay pot over a small fire. She looked up at him with gentle eyes.

“Sitlwa, ngwana,” she murmured softly (Be calm, child). Her voice was quiet but carried the authority of one who has seen much. Kagiso tried to pull himself up. His limbs felt weak. “Rest,” she said, pressing a hand gently on his arm. She held a wooden bowl to his lips. “Drink this. It will help.”

Dazed, Kagiso obeyed. The bitter taste of ginger water slid down his throat, warming his belly. Memory slowly seeped back. Rain-soaked earth, vines, the cry of ghosts in his ears. He winced as he remembered the phantom faces and spears of light. He looked at his hands; they were stained brown with mud and something darker.

“What happened?” he whispered, voice brittle and dry. The woman placed a finger on her lips. “You were very brave,” she said. Her accent was gentle, low. “My spear… my home…” Kagiso croaked. The pain in his arm made the words cut off.

“Not yet,” the woman said, smiling sadly. “First, you live.” She brushed dirt out of his dreadlocks and nodded toward the grasses outside. “They brought you here after the storm. You are safe now.”

“Where...?” Kagiso’s world spun. He tried to sit up and stopped himself; his arm burned. Cold sweat beaded on his forehead. He remembered the ravine, the fight. “The shadow monsters…” The dreamlike battle of the ravine pressed close in his mind.

The woman stiffened. “Monsters?” she echoed, all warmth gone from her eyes. She met his gaze steadily. “You are safe,” she repeated, though her tone hinted at worry. She spilled some of the ginger water into her hands and began to pour it over his arm. Kagiso gasped as the warm liquid mixed with cool rainwater on his skin. He watched as blood and red water ran off, soaking into the cloth.

He tried to speak, but two men’s voices outside interrupted. Heavy footsteps sounded against packed earth. The woman hushed him at once. She pressed Kagiso back onto the mat, her face a mask of sudden concern. Through the thin wall he heard a man’s stern voice: “He’s awake.”

Then the mud wall slid open. Two guards entered the hut. One was an older man with a shaved head and ritual scars; the other was young, lean, with dark eyes and a spear resting in his hand. Both wore leather jerkins dyed ochre and crimson. Kagiso’s heart lurched.

The elderly guard spoke quietly to the young one. Kagiso caught words like “No scratch on him, but eyes wide as the storm,” and “Badimo” and “tokoloshe.” Banele. The lean guard was Banele, he realized—the Kgosi’s son. Kagiso recognized Banele from childhood memories. Once they had splashed in mud and played among cattle, until Banele had sneered and shoved him away. Now the youth gave him a cold stare. A stylized flame was painted on Banele’s chest piece—the mark of the Kgosi’s (chief’s) clan.

“Kagiso,” the older guard said, surprise creeping into his tone. “You’re alive.” He stepped forward. Kagiso felt his spear tremble in his grip and tried to pull himself up. “We heard the storm,” he managed to croak. His throat was raw. “We fought them. The spirits came…”

Banele’s eyes narrowed. “Spirits?” he spat softly. He took a menacing step closer. “You’d have us believe the Tokoloshe didn’t finish you? That our curse failed?” He snarled. There was poison in his words. Kagiso wasn’t sure what the Tokoloshe was, only that every villager had used that name to fill children’s nightmares.

Kagiso’s ribs ached from where he had tumbled in the fight, but he pushed to his elbows. “Who is the Tokoloshe? (evil spirit)” he demanded, voice cracking. The word had a strange weight, as if the night itself hung on it.

Banele sneered. “An ancient demon-child tale,” he said. “Even the worst thing in darkness. We saw it that night in our dreams. They taught us to fear it.” He turned partly away, tightening a leather band on his arm. “Now they want you dead by morning, Kagiso,” he spat. “Let a Tokoloshe walk free? No. If your Badimo didn’t break you, this will.”

“You call me demon-child again,” Kagiso snapped. He threw himself upright. “If I’m anything, I’m the Child of the Comet,” he growled. He stood on unsteady legs, raising his free hand as if to protect himself.

The older guard intervened, stepping between Kagiso and Banele. “Easy now,” he said sharply. Banele scowled and spat in the dirt.

“You were thrown away as a baby under that star,” Banele continued, ignoring the older guard. “Don’t you remember? We wrote you off. Only the fire-star was cursed, not invited.” He raised a fist, but the elder guard grabbed his shoulder.

“How did you survive, Kagiso?” the elder guard asked softly. His deep voice was grim. “Prophecy said you’d be gone. You—child—how did you live?”

Kagiso’s mouth went dry. He nodded faintly, hand unconsciously pressing against the cut on his arm. “I don’t know,” he whispered. “I fought… and the ancestors answered.”

The elder guard’s face went pale. “Badimo…?” he murmured to himself. He turned to Banele, then back to Kagiso. Slowly, he shook his head. “No wonder the chief is furious,” he said.

The two guards exchanged tense glances, then left the hut. Kagiso felt the Sangoma’s hand on his shoulder. He had not noticed it, but she had been standing silently in the corner. Her eyes were full of something like pity. She gave Kagiso a small bow and motioned toward the door. “Let him rest,” she whispered to the guards.

As they left, Kagiso found himself outside the hut under a gray dawn sky. A cold breeze tugged at his soaked clothes. All around, the village was stirring to life: distant drums, crackling fires, warriors arming themselves. He clutched his spear again, knuckles whitening.

The Sangoma stood beside him, calm. In her hand was Kagiso’s amulet, the little carved wooden eye he had worn since infancy. She held it gently. “Your name means ‘light’,” she said softly, examining the intricate rings carved around the pupil. “You carry the Comet’s blood, and the Badimo’s vision. Take this.”

Before Kagiso could react, she pressed the amulet back into his hand. “This was your mother’s. It contains words of strength. Remember them.” She wouldn’t say more. Kagiso slipped the amulet safely under his shirt.

They walked away from the compound’s main square. Even at this early hour, villagers clustered around a glowing fire pit where the dawn ritual would begin. Eight carved wooden banners stood in a circle, each marked with a clan symbol. Kagiso recognized some of them: the roaring flame of the Kgosi’s fire clan, the swirling wave of the Metsi (water) clan of the river, the lightning bolt of the Dikgoka (lightning) clan of thunder, a horned skull of the cattle-keepers, a rising sun of the harvest clan, and others.

Atop a high wooden dais sat the Kgosi himself, draped in dark cloth and smoking a slender pipe. His eyes were red-rimmed. Though he said nothing, the entire village was silent under his gaze. Kagiso met his stare for a moment. The chief’s eyes were the color of embers. They were full of fear, anger, and something like guilt.

Kagiso’s own face burned with shame and anger. He felt raw under so many eyes. The clans watched him like a wilder beast’s rescue. He gripped his spear tightly. The air smelled of smoked meat and sandalwood. Somewhere a rooster crowed — absurdly mundane amid this tension.

Suddenly, from the fire pit a carved wooden bull’s head snapped at Kagiso’s ankles and winked shut. Its eyes glowed. He jumped back with a gasp as a faint click echoed; the trap-door was locked again. On its brass ring was etched a single word: Tokoloho (freedom). Kagiso swallowed. Tokoloho.

The Sangoma gave him a small smile and patted his arm. “The omen is yours now,” she said softly. “The Child of the Comet walks again.” She added quietly, “But at what price, ngwana (child)?”

Kagiso said nothing. He looked up at the comet, its pale streak high above. In its tail he thought he saw the faces of his lost family, urging him on. Lightning flashed, as if the dawn itself answered. A chill breeze moved through the village.

He turned back to find the Sangoma already retreating into the shadows of the huts. Behind him, on the roof of the incense tower, two crows let out a long caw. Kagiso clenched the spear and took a deep breath. Though the night’s battle was done, he realized it was only the beginning of what he must face.

In the silent fields beyond the village, two pale red eyes blinked at the falling morning. The Tokoloshe watched and waited as the comet’s light faded into day."

output_dir = Path('outputs')
story_title_slug = "Shadow_of_the_comet_Chapter_1"
audio_file_path = str(output_dir / f"{story_title_slug}_Audiobook.wav") 
if __name__ == "__main__":
    try:
        audio_file = generate_audio(audio_text, "shadow of the comet" , audio_file_path) 
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.error(f"Script exited with a critical failure.")
        # The logger already captured the detailed exception, so we can exit.
        exit(1)
    finally:
        # Ensure the Telegram logger is closed gracefully
        if 'telegram_handler' in globals() and telegram_handler is not None:
            telegram_handler.close()

