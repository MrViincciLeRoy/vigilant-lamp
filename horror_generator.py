import logging
from pathlib import Path
import re
from typing import Dict, List
import gc
from llama_cpp import Llama
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import kokoro 
import os
import requests
import json
from datetime import datetime
import threading
import queue
import time

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')  # Replace with your bot token
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')    # Replace with your chat ID
def send_audio_file_to_telegram(bot_token: str, chat_id: str, audio_file_path: str, title: str):
    """Send the audio file to Telegram with upload progress."""
    try:
        logger.info(f"Preparing to send audio file to Telegram: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False

        # Get file size for progress tracking
        file_size = os.path.getsize(audio_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")

        # Check if file is too large (Telegram limit is 50MB for bots)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"File too large for Telegram ({file_size_mb:.2f} MB). Maximum is 50MB.")
            
            # Send notification about file being too large
            message = f"🎵 *Audio File Ready*\n\n"
            message += f"*Title:* {title}\n"
            message += f"*Size:* {file_size_mb:.2f} MB\n"
            message += f"*Status:* ❌ File too large for Telegram (max 50MB)\n"
            message += f"*Location:* {audio_file_path}\n"
            message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            requests.post(url, json=payload, timeout=10)
            return False

        # Send upload start notification
        start_message = f"🎵 *Uploading Audio File*\n\n"
        start_message += f"*Title:* {title}\n"
        start_message += f"*Size:* {file_size_mb:.2f} MB\n"
        start_message += f"*Status:* Uploading to Telegram...\n"
        start_message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': start_message,
            'parse_mode': 'Markdown'
        }
        requests.post(url, json=payload, timeout=10)

        # Upload the audio file
        url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'audio': (os.path.basename(audio_file_path), audio_file, 'audio/wav')
            }
            
            data = {
                'chat_id': chat_id,
                'title': title,
                'caption': f"🎵 *{title}* - Horror Story Audiobook\n\n📊 File size: {file_size_mb:.2f} MB\n⏱️ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'parse_mode': 'Markdown'
            }

            logger.info("Uploading audio file to Telegram...")
            
            # Use a longer timeout for large file uploads
            timeout = max(300, int(file_size_mb * 10))  # At least 5 minutes, more for larger files
            response = requests.post(url, files=files, data=data, timeout=timeout)

            if response.ok:
                logger.info("✅ Audio file successfully sent to Telegram!")
                return True
            else:
                logger.error(f"Failed to send audio file: {response.status_code} - {response.text}")
                
                # Send error notification
                error_message = f"🎵 *Audio Upload Failed*\n\n"
                error_message += f"*Title:* {title}\n"
                error_message += f"*Error:* {response.status_code} - Upload failed\n"
                error_message += f"*File Location:* {audio_file_path}\n"
                error_message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"

                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': error_message,
                    'parse_mode': 'Markdown'
                }
                requests.post(url, json=payload, timeout=10)
                return False

    except requests.exceptions.Timeout:
        logger.error("Timeout while uploading audio file to Telegram")
        
        # Send timeout notification
        timeout_message = f"🎵 *Audio Upload Timeout*\n\n"
        timeout_message += f"*Title:* {title}\n"
        timeout_message += f"*Status:* Upload timed out (file may be too large)\n"
        timeout_message += f"*File Location:* {audio_file_path}\n"
        timeout_message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': timeout_message,
            'parse_mode': 'Markdown'
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except:
            pass
        return False
        
    except Exception as e:
        logger.error(f"Error sending audio file to Telegram: {e}")
        return False
class TelegramLogHandler(logging.Handler):
    """Custom logging handler that sends logs to Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.message_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.start_worker()
    
    def start_worker(self):
        """Start the background worker thread for sending messages."""
        self.worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self.worker_thread.start()
    
    def _message_worker(self):
        """Background worker that sends queued messages to Telegram."""
        while not self.stop_event.is_set():
            try:
                # Get message from queue with timeout
                message = self.message_queue.get(timeout=1.0)
                self._send_to_telegram(message)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in Telegram worker: {e}")
    
    def _send_to_telegram(self, message: str):
        """Send a message to Telegram."""
        try:
            # Telegram has a 4096 character limit, so we need to split long messages
            max_length = 4000  # Leave some buffer
            
            if len(message) <= max_length:
                messages = [message]
            else:
                # Split message into chunks
                messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            
            for msg in messages:
                payload = {
                    'chat_id': self.chat_id,
                    'text': msg,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(self.url, json=payload, timeout=10)
                
                if not response.ok:
                    print(f"Failed to send Telegram message: {response.status_code} - {response.text}")
                
                # Rate limiting - Telegram allows 30 messages per second
                time.sleep(0.1)
                
        except requests.exceptions.RequestException as e:
            print(f"Network error sending to Telegram: {e}")
        except Exception as e:
            print(f"Unexpected error sending to Telegram: {e}")
    
    def emit(self, record):
        """Handle a log record by sending it to Telegram."""
        try:
            # Format the log message
            log_entry = self.format(record)
            
            # Add emoji based on log level
            emoji_map = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }
            
            emoji = emoji_map.get(record.levelname, 'ℹ️')
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Format message for Telegram
            telegram_message = f"{emoji} *{record.levelname}* `{timestamp}`\n```\n{log_entry}\n```"
            
            # Queue the message for sending
            self.message_queue.put(telegram_message)
            
        except Exception as e:
            print(f"Error formatting log for Telegram: {e}")
    
    def close(self):
        """Clean shutdown of the handler."""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        super().close()

def setup_telegram_logging(bot_token: str, chat_id: str):
    """Set up Telegram logging integration."""
    try:
        # Test Telegram connection first
        test_url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(test_url, timeout=10)
        
        if not response.ok:
            print(f"Failed to connect to Telegram bot: {response.status_code}")
            return None
        
        bot_info = response.json()
        print(f"Connected to Telegram bot: {bot_info.get('result', {}).get('username', 'Unknown')}")
        
        # Create and configure the Telegram handler
        telegram_handler = TelegramLogHandler(bot_token, chat_id)
        telegram_handler.setLevel(logging.INFO)  # Only send INFO and above to Telegram
        
        # Create a formatter for Telegram messages
        telegram_formatter = logging.Formatter('%(name)s - %(message)s')
        telegram_handler.setFormatter(telegram_formatter)
        
        return telegram_handler
        
    except requests.exceptions.RequestException as e:
        print(f"Network error connecting to Telegram: {e}")
        return None
    except Exception as e:
        print(f"Error setting up Telegram logging: {e}")
        return None

def send_telegram_progress(bot_token: str, chat_id: str, title: str, progress: str, details: str = ""):
    """Send a formatted progress update to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        message = f"📖 *Story Generation Progress*\n\n"
        message += f"*Title:* {title}\n"
        message += f"*Status:* {progress}\n"
        
        if details:
            message += f"*Details:* {details}\n"
        
        message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"
        
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        requests.post(url, json=payload, timeout=10)
        
    except Exception as e:
        print(f"Error sending progress update: {e}")

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Telegram logging if tokens are configured
telegram_handler = None
if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
    telegram_handler = setup_telegram_logging(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    if telegram_handler:
        logger.addHandler(telegram_handler)
        logger.info("🚀 Telegram logging enabled - Horror story generator starting!")
    else:
        logger.warning("Failed to enable Telegram logging - continuing with console only")
else:
    logger.info("Telegram tokens not configured - using console logging only")

# Load the Llama model
try:
    logger.info("Loading Llama model...")
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=2048,  # Context window size
        n_batch=128,  # Batch size for processing
        n_threads=None,
        n_gpu_layers=0,  # Run on CPU
        verbose=False
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def extract_story_idea_from_prompt(prompt_text: str) -> Dict[str, str]:
    """Extracts story elements from a prompt."""
    idea = {'title': 'Untitled Horror Story'}
    lines = prompt_text.split('\n')
    for line in lines:
        if line.startswith('# '):
            idea['title'] = line[2:].strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_').replace('-', '_')
            idea[key] = value.strip()
    return idea

class SuburbanHorrorGenerator:
    """Generates 'Girl Next Door' style psychological horror stories with a six-act structure."""
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.section_summaries = []
        self.current_story_data = {}
        self.total_target_words = 7200  # Total words for the story
        self.health_threshold = 0.75    # Minimum word count ratio

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        """Defines a six-act story structure with four sections per act for suburban psychological horror."""
        protagonist = story_idea.get('protagonist', 'a curious teenage boy')
        antagonist = story_idea.get('antagonist', 'seemingly normal neighbors')
        setting = story_idea.get('setting', 'a quiet suburban neighborhood')
        victim = story_idea.get('victim', 'the girl next door')
        secret = story_idea.get('secret', 'a horrifying truth hidden in plain sight')

        acts = [
            {
                "act": 1, "title": "Normal Life", "description": "Establishes the ordinary suburban setting and characters.",
                "sections": [
                    {"section": 1, "title": "The Neighborhood", "description": f"Introduction to {setting} where {protagonist} lives a typical suburban life.", "target_words": 300},
                    {"section": 2, "title": "Meeting the Girl", "description": f"{protagonist} encounters {victim} and is intrigued by her.", "target_words": 300},
                    {"section": 3, "title": "Friendly Neighbors", "description": f"The {antagonist} appear to be model citizens in the community.", "target_words": 300},
                    {"section": 4, "title": "First Curiosity", "description": f"{protagonist} begins to notice small, odd details about the house next door.", "target_words": 300}
                ]
            },
            {
                "act": 2, "title": "Growing Suspicion", "description": "Strange observations pile up.",
                "sections": [
                    {"section": 1, "title": "Unusual Sounds", "description": f"{protagonist} hears disturbing noises coming from the neighbors' house at night.", "target_words": 300},
                    {"section": 2, "title": "The Girl's Behavior", "description": f"{victim} shows signs of fear and distress that seem out of place.", "target_words": 300},
                    {"section": 3, "title": "Adult Dismissal", "description": f"Adults refuse to believe {protagonist}'s concerns about the neighbors.", "target_words": 300},
                    {"section": 4, "title": "Investigating Alone", "description": f"{protagonist} decides to investigate the strange occurrences independently.", "target_words": 300}
                ]
            },
            {
                "act": 3, "title": "Horrible Discovery", "description": "The truth begins to emerge.",
                "sections": [
                    {"section": 1, "title": "Peeping In", "description": f"{protagonist} witnesses something disturbing through a window or basement.", "target_words": 300},
                    {"section": 2, "title": "The Secret Room", "description": f"Discovery of a hidden space where {secret} is partially revealed.", "target_words": 300},
                    {"section": 3, "title": "Moral Paralysis", "description": f"{protagonist} struggles with the horror of what they've seen and what to do.", "target_words": 300},
                    {"section": 4, "title": "Failed Attempt", "description": f"An initial attempt to help {victim} goes wrong or is thwarted.", "target_words": 300}
                ]
            },
            {
                "act": 4, "title": "Deeper Into Hell", "description": "The full scope of the horror is revealed.",
                "sections": [
                    {"section": 1, "title": "The Complete Truth", "description": f"The full extent of what {antagonist} are doing to {victim} is revealed.", "target_words": 300},
                    {"section": 2, "title": "Community Complicity", "description": f"Other neighbors know about {secret} but choose to ignore it.", "target_words": 300},
                    {"section": 3, "title": "Escalating Torture", "description": f"The abuse of {victim} becomes more severe and systematic.", "target_words": 300},
                    {"section": 4, "title": "Trapped", "description": f"{protagonist} realizes they may be in danger for knowing too much.", "target_words": 300}
                ]
            },
            {
                "act": 5, "title": "Breaking Point", "description": "Everything comes to a head.",
                "sections": [
                    {"section": 1, "title": "Desperate Plan", "description": f"{protagonist} devises a risky plan to save {victim}.", "target_words": 300},
                    {"section": 2, "title": "The Confrontation", "description": f"Direct confrontation with {antagonist} over their treatment of {victim}.", "target_words": 300},
                    {"section": 3, "title": "Violence Erupts", "description": f"The suburban facade crumbles as violence breaks out.", "target_words": 300},
                    {"section": 4, "title": "Life or Death", "description": f"A final struggle determines the fate of both {protagonist} and {victim}.", "target_words": 300}
                ]
            },
            {
                "act": 6, "title": "Aftermath", "description": "The story concludes with lasting consequences.",
                "sections": [
                    {"section": 1, "title": "The Reckoning", "description": f"Authorities finally get involved and the truth comes to light.", "target_words": 300},
                    {"section": 2, "title": "Damaged Lives", "description": f"The psychological toll on {protagonist} and {victim} is revealed.", "target_words": 300},
                    {"section": 3, "title": "Suburban Scars", "description": f"The neighborhood is forever changed by the revelation of {secret}.", "target_words": 300},
                    {"section": 4, "title": "Haunted Memory", "description": f"The lasting impact of the horror on all who survived it.", "target_words": 300}
                ]
            }
        ]
        return acts

    def _build_focused_context(self) -> str:
        """
        Builds comprehensive context with characters, plot, and recent text to maintain consistency.
        """
        # Persistent character context
        protagonist = self.current_story_data.get('protagonist', 'A curious teenage boy')
        antagonist = self.current_story_data.get('antagonist', 'Seemingly normal neighbors')
        victim = self.current_story_data.get('victim', 'The girl next door')
        setting = self.current_story_data.get('setting', 'A quiet suburban neighborhood')

        char_context = (
            "KEY CHARACTERS (maintain these throughout):\n"
            f"- Protagonist: {protagonist}\n"
            f"- Antagonist: {antagonist}\n"
            f"- Victim: {victim}\n"
            f"- Setting: {setting}\n"
        )

        # Include more recent context for better continuity
        plot_context = ""
        if self.section_summaries:
            plot_context = "STORY PROGRESSION:\n" + "\n".join(self.section_summaries)
        
        # Include the last written section to maintain narrative flow
        recent_text = ""
        if hasattr(self, 'previous_section_text') and self.previous_section_text:
            # Get last few sentences for context
            sentences = self.previous_section_text.split('.')
            if len(sentences) > 2:
                last_sentences = '. '.join(sentences[-3:]).strip()
                recent_text = f"\nLAST WRITTEN TEXT (continue from here):\n...{last_sentences}"

        return f"{char_context}\n{plot_context}{recent_text}"

    def generate_section(self, act_num: int, section_num: int, section_data: Dict[str, str]) -> str:
        """Generates a single story section with retry logic if word count is insufficient."""
        if not self.model_loaded:
            return "[Error: Model not loaded]"
    
        title = section_data["title"]
        description = section_data["description"]
        target_words = section_data["target_words"]
    
        # Send progress update to Telegram
        if telegram_handler:
            story_title = self.current_story_data.get('title', 'Unknown Story')
            progress_msg = f"Writing Act {act_num}, Section {section_num}: {title}"
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, story_title, progress_msg, f"Target: {target_words} words")
    
        horror_style_guide = (
            "WRITING STYLE: Suburban Psychological Horror (like 'The Girl Next Door'):\n"
            "- Focus on the horror hidden beneath suburban normalcy.\n"
            "- Emphasize psychological tension over gore.\n"
            "- Show the complicity of 'normal' people in evil.\n"
            "- Build dread through mundane details that feel wrong.\n"
            "- Explore themes of innocence lost and moral cowardice.\n"
            "- Use realistic dialogue and believable character motivations.\n"
            "- Show how evil can flourish when good people do nothing."
        )
    
        context = self._build_focused_context()
    
        prompt = f"""
    You are continuing a psychological horror story in the style of "The Girl Next Door". This must flow seamlessly from the previous section.
    
    STORY TITLE: "{self.current_story_data['title']}"
    {context}
    
    CURRENT SECTION: Act {act_num}, Section {section_num} - {title}
    SECTION GOAL: {description}
    
    {horror_style_guide}
    
    CRITICAL INSTRUCTIONS:
    - Write ~{target_words} words that flow DIRECTLY from where the last section ended.
    - If there was previous text, continue the narrative smoothly - don't restart or summarize.
    - Maintain the same characters, their personalities, and the established tone.
    - Keep the same POV and narrative voice throughout.
    - Advance the plot toward the section goal while maintaining story flow.
    - NO section headers, titles, or meta-commentary in the story text.
    - Write as if this is one continuous story, not separate sections.
    - Begin writing the continuation now:
    """
    
        logger.info(f"Generating Act {act_num}, Section {section_num}: '{title}'...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                output = self.llm(
                    prompt,
                    max_tokens=int(target_words * 1.3),  # Slightly more tokens for better completion
                    temperature=0.5,  # Lower temperature for more consistent narrative
                    top_p=0.85,  # More focused generation
                    repeat_penalty=1.1,  # Less aggressive to maintain flow
                    stop=["\n\n\n\n", "---", "## Act", "### Section", "CHAPTER", "THE END"],
                    echo=False
                )
                section_text = self._clean_text(output['choices'][0]['text'])
    
                if self._validate_section(section_text, target_words):
                    word_count = len(section_text.split())
                    logger.info(f"  ✓ Act {act_num}, Section {section_num} completed ({word_count} words) after {attempt + 1} attempt(s).")
                    self.previous_section_text = section_text
                    summary = f"Act {act_num}-{section_num} ({title}): {description[:100]}..."
                    self.section_summaries.append(summary)
                    return section_text
                else:
                    word_count = len(section_text.split())
                    logger.warning(f"  ✗ Attempt {attempt + 1}: Section too short ({word_count} words), retrying...")
            except Exception as e:
                logger.error(f"  ✗ Error during attempt {attempt + 1}: {e}")
    
        logger.error(f"Failed to generate Act {act_num}, Section {section_num} after {max_attempts} attempts")
        return f"[Error: Could not generate Act {act_num}, Section {section_num}]"

    def _clean_text(self, text: str) -> str:
        """
        Removes unwanted lines and artifacts, and attempts to fix encoding errors.
        """
        # Attempt to fix common encoding errors from model output
        try:
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        text = text.strip()

        # Remove unwanted headers, instructions, and AI self-prompts
        lines = text.split('\n')
        unwanted_patterns = re.compile(
            r'^(WRITING STYLE|INSTRUCTIONS|SECTION|You are|STORY TITLE|CURRENT SECTION|SECTION GOAL|PREVIOUSLY IN THE STORY|KEY CHARACTERS)|'
            r'^(Word Count: \d+)|'
            r'^(Please critique this section)|'
            r'^---$',
            re.IGNORECASE
        )
        cleaned_lines = [line for line in lines if not unwanted_patterns.match(line.strip())]

        return '\n\n'.join(cleaned_lines).strip()

    def _validate_section(self, section_text: str, target_words: int) -> bool:
        """Checks if the section meets word count requirements."""
        if not section_text or section_text.startswith("[Error"):
            return False
        word_count = len(section_text.split())
        if word_count < target_words * self.health_threshold:
            logger.warning(f"    - Validation failed: Word count ({word_count}) too low.")
            return False
        return True

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        """Generates the full six-act story."""
        if not self.model_loaded:
            raise ConnectionError("Model not loaded.")

        logger.info(f"\n=== Generating Story: '{story_idea['title']}' ===")
        self.current_story_data = story_idea
        self.section_summaries = []
        self.previous_section_text = ""  # Initialize for continuity

        # Send start notification to Telegram
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, story_idea['title'], "Starting story generation", "6 acts, 24 sections total")

        story_structure = self.create_story_structure(story_idea)
        full_story_sections = []
        total_words = 0

        for act in story_structure:
            act_num = act['act']
            act_title = act['title']
            
            logger.info(f"\n--- Starting Act {act_num}: {act_title} ---")
            
            # Only add act headers, not section headers for smoother flow
            if act_num == 1:
                # Start the story without any headers
                pass
            else:
                # Add minimal act breaks for longer stories
                full_story_sections.append(f"\n\n* * *\n\n")
            
            for section in act['sections']:
                section_num = section['section']
                section_title = section['title']
                section_text = self.generate_section(act_num, section_num, section)
                if not section_text.startswith("[Error"):
                    # Add section text without headers to maintain flow
                    full_story_sections.append(f"{section_text}\n\n")
                    total_words += len(section_text.split())
                else:
                    full_story_sections.append(f"_{section_text}_\n\n")
                gc.collect()

        complete_story_text = "".join(full_story_sections)

        logger.info("\n=== Story Generation Complete ===")
        logger.info(f"Final Word Count: {total_words} / {self.total_target_words}")

        # Send completion notification to Telegram
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, story_idea['title'], "✅ Story generation completed!", f"Final word count: {total_words}")

        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words,
        }

def clean_for_audio(text: str) -> str:
    """
    Removes Markdown formatting for clean audio generation.
    """
    logger.info("Cleaning text for audio by removing Markdown...")
    # Remove markdown headers (e.g., ##, ###) and surrounding whitespace
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown bolding (**)
    text = text.replace('**', '')
    # Remove markdown horizontal rules (---) and other separators
    text = re.sub(r'^\s*---\s*$', '', text, flags=re.MULTILINE)
    # Normalize newlines to ensure consistent spacing
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    logger.info("Text cleaning complete.")
    return text

def generate_audio(text: str, output_file: str = 'outputs/complete_narration.wav'):
    """Generate audio narration from clean text."""
    try:
        logger.info("Generating audio narration...")
        
        # Send audio generation start notification
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, "Audio Generation", "Starting audio narration", "Converting text to speech")
        
        os.makedirs('outputs', exist_ok=True)

        pipeline = KPipeline(lang_code='a')
        audio_segments = []

        # Split text into manageable chunks by paragraphs
        text_chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]

        total_chunks = len(text_chunks)
        logger.info(f"Splitting text into {total_chunks} chunks for audio processing.")

        for i, chunk in enumerate(text_chunks):
            if chunk:
                logger.info(f"Processing audio for chunk {i+1}/{total_chunks}...")
                
                # Send periodic progress updates to Telegram
                if telegram_handler and i % 10 == 0:  # Every 10 chunks
                    progress = f"Processing chunk {i+1}/{total_chunks}"
                    send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, "Audio Generation", progress, f"{(i/total_chunks)*100:.1f}% complete")
                
                generator = pipeline(chunk, voice='bm_fable')
                for _, _, audio in generator:
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)

        if audio_segments:
            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            logger.info(f"Audio saved as '{output_file}' (Duration: {duration:.2f} seconds)")
            
            # Send audio completion notification
            if telegram_handler:
                send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, "Audio Generation", "✅ Audio generation completed!", f"Duration: {duration:.2f} seconds")
            
            return output_file
        else:
            logger.warning("No audio segments were generated.")
            return None

    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        return None
        
def main():
    """Main function to generate story and audio."""
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    story_prompt = """
# The House on Maple Street

Protagonist: David Parker, a 16-year-old boy who recently moved to Millbrook with his family. He's observant, curious, and struggles to fit in at his new school.
Antagonist: Ruth and Richard Chandler, a middle-aged couple who appear to be model citizens - she volunteers at church, he coaches little league. They live in the immaculate house next door.
Setting: Millbrook, a picturesque suburban town where everyone knows everyone, lawns are perfectly manicured, and dark secrets hide behind white picket fences.
Victim: Meg Loughlin, a 14-year-old girl who lives with the Chandlers as their "niece" after her parents died in a car accident. She's quiet, withdrawn, and always seems afraid.
Secret: The Chandlers are systematically torturing and abusing Meg, while the entire neighborhood either ignores the signs or actively covers it up to maintain their perfect community image.
"""

    try:
        logger.info("Extracting story idea...")
        story_idea = extract_story_idea_from_prompt(story_prompt)
        logger.info(f"Story idea extracted for title: '{story_idea.get('title', 'N/A')}'")

        generator = SuburbanHorrorGenerator()
        logger.info("Generating story...")
        final_story = generator.generate_complete_story(story_idea)

        if final_story and final_story['word_count'] > 0:
            story_title = final_story['title'].replace(' ', '_')

            # Save the original markdown file
            story_file_md = output_dir / f"{story_title}.md"
            with open(story_file_md, 'w', encoding='utf-8') as f:
                f.write(f"# {final_story['title']}\n\n")
                f.write(f"**Word Count:** {final_story['word_count']}\n\n")
                f.write(final_story['text'])
            logger.info(f"Story saved to {story_file_md}")

            # Clean the text for audio generation
            audio_text = clean_for_audio(final_story['text'])

            # Save a clean .txt file for verification
            clean_text_file = output_dir / f"{story_title}_Clean.txt"
            with open(clean_text_file, 'w', encoding='utf-8') as f:
                f.write(audio_text)
            logger.info(f"Clean text for audio verification saved to {clean_text_file}")

            # Generate audio from the clean text
            audio_file_path = str(output_dir / f"{story_title}_Audiobook.wav")
            audio_file = generate_audio(audio_text, audio_file_path)

            # Send audio file via Telegram if generated successfully
            if audio_file and telegram_handler:
                logger.info(f"Audio file generated: {audio_file}")
                send_audio_file_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, audio_file, final_story['title'])

            # Send final completion message
            if telegram_handler:
                send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_story['title'], "🎉 All tasks completed!", f"Story: {final_story['word_count']} words, Audio generated")

        else:
            logger.error("Story generation failed: empty content returned.")

    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with critical error: {e}")
        exit(1)