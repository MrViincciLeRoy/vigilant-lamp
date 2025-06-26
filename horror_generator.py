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
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_audio_file_to_telegram(bot_token: str, chat_id: str, audio_file_path: str, title: str):
    """Send the audio file to Telegram with upload progress."""
    try:
        logger.info(f"Preparing to send audio file to Telegram: {audio_file_path}")
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False
        file_size = os.path.getsize(audio_file_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")
        if file_size > 50 * 1024 * 1024:
            logger.warning(f"File too large for Telegram ({file_size_mb:.2f} MB). Maximum is 50MB.")
            message = f"🎵 *Audio File Ready*\n\n*Title:* {title}\n*Size:* {file_size_mb:.2f} MB\n*Status:* ❌ File too large\n*Location:* {audio_file_path}\n*Time:* {datetime.now().strftime('%H:%M:%S')}"
            requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}, timeout=10)
            return False
        start_message = f"🎵 *Uploading Audio File*\n\n*Title:* {title}\n*Size:* {file_size_mb:.2f} MB\n*Status:* Uploading...\n*Time:* {datetime.now().strftime('%H:%M:%S')}"
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={'chat_id': chat_id, 'text': start_message, 'parse_mode': 'Markdown'}, timeout=10)
        url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': (os.path.basename(audio_file_path), audio_file, 'audio/wav')}
            data = {'chat_id': chat_id, 'title': title, 'caption': f"🎵 *{title}* - Horror Story Audiobook\n\n📊 File size: {file_size_mb:.2f} MB\n⏱️ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'parse_mode': 'Markdown'}
            timeout = max(300, int(file_size_mb * 10))
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.ok:
                logger.info("✅ Audio file successfully sent to Telegram!")
                return True
            else:
                logger.error(f"Failed to send audio file: {response.status_code} - {response.text}")
                error_message = f"🎵 *Audio Upload Failed*\n\n*Title:* {title}\n*Error:* {response.status_code} - Upload failed\n*File Location:* {audio_file_path}\n*Time:* {datetime.now().strftime('%H:%M:%S')}"
                requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={'chat_id': chat_id, 'text': error_message, 'parse_mode': 'Markdown'}, timeout=10)
                return False
    except requests.exceptions.Timeout:
        logger.error("Timeout while uploading audio file to Telegram")
        timeout_message = f"🎵 *Audio Upload Timeout*\n\n*Title:* {title}\n*Status:* Upload timed out\n*File Location:* {audio_file_path}\n*Time:* {datetime.now().strftime('%H:%M:%S')}"
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={'chat_id': chat_id, 'text': timeout_message, 'parse_mode': 'Markdown'}, timeout=10)
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
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self.worker_thread.start()

    def _message_worker(self):
        while not self.stop_event.is_set():
            try:
                message = self.message_queue.get(timeout=1.0)
                max_length = 4000
                messages = [message[i:i+max_length] for i in range(0, len(message), max_length)] if len(message) > max_length else [message]
                for msg in messages:
                    requests.post(self.url, json={'chat_id': self.chat_id, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
                    time.sleep(0.1)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in Telegram worker: {e}")

    def emit(self, record):
        try:
            log_entry = self.format(record)
            emoji_map = {'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌', 'CRITICAL': '🚨'}
            emoji = emoji_map.get(record.levelname, 'ℹ️')
            timestamp = datetime.now().strftime('%H:%M:%S')
            telegram_message = f"{emoji} *{record.levelname}* `{timestamp}`\n```\n{log_entry}\n```"
            self.message_queue.put(telegram_message)
        except Exception as e:
            print(f"Error formatting log for Telegram: {e}")

    def close(self):
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        super().close()

def setup_telegram_logging(bot_token: str, chat_id: str):
    try:
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
        if not response.ok:
            print(f"Failed to connect to Telegram bot: {response.status_code}")
            return None
        bot_info = response.json()
        print(f"Connected to Telegram bot: {bot_info.get('result', {}).get('username', 'Unknown')}")
        telegram_handler = TelegramLogHandler(bot_token, chat_id)
        telegram_handler.setLevel(logging.INFO)
        telegram_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
        return telegram_handler
    except Exception as e:
        print(f"Error setting up Telegram logging: {e}")
        return None

def send_telegram_progress(bot_token: str, chat_id: str, title: str, progress: str, details: str = ""):
    try:
        message = f"📖 *Story Generation Progress*\n\n*Title:* {title}\n*Status:* {progress}\n"
        if details:
            message += f"*Details:* {details}\n"
        message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}"
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"Error sending progress update: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

try:
    logger.info("Loading Llama model...")
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=8192,  # Increased to 8k context
        n_batch=128,
        n_threads=None,
        n_gpu_layers=0,
        verbose=False
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def extract_story_idea_from_prompt(prompt_text: str) -> Dict[str, str]:
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
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.section_summaries = []
        self.current_story_data = {}
        self.total_target_words = 7200  # 6 acts × 2 sections × 600 words
        self.health_threshold = 0.75
        self.previous_section_text = ""

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        protagonist = story_idea.get('protagonist', 'a curious teenage boy')
        antagonist = story_idea.get('antagonist', 'seemingly normal neighbors')
        setting = story_idea.get('setting', 'a quiet suburban neighborhood')
        victim = story_idea.get('victim', 'the girl next door')
        secret = story_idea.get('secret', 'a horrifying truth hidden in plain sight')
        
        acts = [
            {"act": 1, "title": "Normal Life", "description": "Establishes the ordinary suburban setting and characters.",
             "sections": [
                 {"section": 1, "title": "The Neighborhood", "description": f"Introduction to {setting} where {protagonist} lives a typical suburban life. Establish the peaceful facade and introduce key locations.", "target_words": 600},
                 {"section": 2, "title": "Meeting the Girl", "description": f"{protagonist} encounters {victim} and the {antagonist}. Show the surface-level normalcy while hinting at something beneath.", "target_words": 600}
             ]},
            {"act": 2, "title": "Growing Suspicion", "description": "Strange observations pile up.",
             "sections": [
                 {"section": 1, "title": "Unusual Signs", "description": f"{protagonist} begins noticing odd details - sounds, behaviors, inconsistencies in the neighbors' perfect facade.", "target_words": 600},
                 {"section": 2, "title": "Adult Dismissal", "description": f"Adults refuse to believe {protagonist}'s concerns. The isolation begins as {victim} shows more signs of distress.", "target_words": 600}
             ]},
            {"act": 3, "title": "Horrible Discovery", "description": "The truth begins to emerge.",
             "sections": [
                 {"section": 1, "title": "First Glimpse", "description": f"{protagonist} witnesses something disturbing that confirms their worst fears about what's happening to {victim}.", "target_words": 600},
                 {"section": 2, "title": "Moral Paralysis", "description": f"{protagonist} struggles with the horror of discovery and attempts to help, but faces the reality of how deep the secret goes.", "target_words": 600}
             ]},
            {"act": 4, "title": "Deeper Into Hell", "description": "The full scope of the horror is revealed.",
             "sections": [
                 {"section": 1, "title": "The Complete Truth", "description": f"The full extent of {secret} is revealed. {protagonist} learns how systematic and widespread the abuse truly is.", "target_words": 600},
                 {"section": 2, "title": "Community Complicity", "description": f"The shocking realization that others know about {secret} but choose silence. {protagonist} realizes they may be in danger.", "target_words": 600}
             ]},
            {"act": 5, "title": "Breaking Point", "description": "Everything comes to a head.",
             "sections": [
                 {"section": 1, "title": "Desperate Plan", "description": f"{protagonist} devises a risky plan to save {victim}, knowing the consequences of failure.", "target_words": 600},
                 {"section": 2, "title": "The Confrontation", "description": f"The plan unfolds leading to direct confrontation with {antagonist}. Violence erupts as the suburban facade finally crumbles.", "target_words": 600}
             ]},
            {"act": 6, "title": "Aftermath", "description": "The story concludes with lasting consequences.",
             "sections": [
                 {"section": 1, "title": "The Reckoning", "description": f"The immediate aftermath of the confrontation. Authorities get involved and the truth begins to come to light.", "target_words": 600},
                 {"section": 2, "title": "Haunted Memory", "description": f"The lasting psychological impact on {protagonist}, {victim}, and the community. The neighborhood is forever changed.", "target_words": 600}
             ]}
        ]
        return acts

    def _build_comprehensive_context(self) -> str:
        """Build detailed context for story consistency."""
        protagonist = self.current_story_data.get('protagonist', 'A curious teenage boy')
        antagonist = self.current_story_data.get('antagonist', 'Seemingly normal neighbors')
        victim = self.current_story_data.get('victim', 'The girl next door')
        setting = self.current_story_data.get('setting', 'A quiet suburban neighborhood')
        secret = self.current_story_data.get('secret', 'A horrifying truth hidden in plain sight')
        
        char_context = f"""CHARACTER CONSISTENCY (maintain exactly):
- Protagonist: {protagonist}
- Antagonist: {antagonist}  
- Victim: {victim}
- Setting: {setting}
- Central Secret: {secret}"""
        
        plot_context = ""
        if self.section_summaries:
            plot_context = f"\nSTORY PROGRESSION SO FAR:\n" + "\n".join(self.section_summaries)
        
        recent_context = ""
        if self.previous_section_text:
            # Get last 200 words for better continuity
            words = self.previous_section_text.split()
            if len(words) > 50:
                recent_words = words[-200:] if len(words) >= 200 else words[-50:]
                recent_context = f"\nDIRECT CONTINUATION FROM PREVIOUS SECTION:\n...{' '.join(recent_words)}"
        
        return f"{char_context}\n{plot_context}{recent_context}"

    def generate_section(self, act_num: int, section_num: int, section_data: Dict[str, str], retry_count: int = 0, max_retries: int = 3) -> str:
        if not self.model_loaded:
            return "[Error: Model not loaded]"
        
        title = section_data["title"]
        description = section_data["description"]
        target_words = section_data["target_words"]
        
        horror_style_guide = (
            "WRITING STYLE: Suburban Psychological Horror:\n"
            "- Focus on horror beneath suburban normalcy\n"
            "- Emphasize psychological tension over gore\n"
            "- Show complicity of 'normal' people\n"
            "- Build dread with mundane details that feel wrong\n"
            "- Explore themes of innocence lost and moral cowardice\n"
            "- Use realistic dialogue and motivations\n"
            "- Show how evil flourishes when good people do nothing"
        )
        
        context = self._build_comprehensive_context()
        
        prompt = f"""Continue this psychological horror story seamlessly from where it left off. Maintain absolute consistency with all established characters, settings, and plot points.

STORY TITLE: "{self.current_story_data['title']}"

{context}

CURRENT SECTION: Act {act_num}, Section {section_num} - {title}
SECTION GOAL: {description}

{horror_style_guide}

CRITICAL INSTRUCTIONS:
- Write exactly as a continuation - no summaries or repetition
- Maintain identical character names, personalities, and relationships
- Keep consistent tone, POV, and writing style
- Advance the plot toward the section goal
- No headers, titles, or meta-commentary
- Begin writing the story continuation now:
"""

        logger.info(f"Generating Act {act_num}, Section {section_num}: '{title}' (Attempt {retry_count + 1}/{max_retries})")
        
        try:
            output = self.llm(
                prompt,
                max_tokens=int(target_words * 2.0),  # Increased multiplier for 600 words
                temperature=0.5,
                top_p=0.85,
                repeat_penalty=1.1,
                stop=["\n\n\n\n", "---", "## Act", "### Section", "CHAPTER", "THE END", "***"],
                echo=False
            )
            
            section_text = self._clean_text(output['choices'][0]['text'])
            
            if self._validate_section(section_text, target_words):
                word_count = len(section_text.split())
                logger.info(f"  ✓ Act {act_num}, Section {section_num} completed - {word_count} words")
                
                # Store for next section's context
                self.previous_section_text = section_text
                
                # Add to story progression summary
                summary = f"Act {act_num}-{section_num} ({title}): {description[:80]}..."
                self.section_summaries.append(summary)
                
                return section_text
            else:
                word_count = len(section_text.split())
                logger.warning(f"  ✗ Act {act_num}, Section {section_num} validation failed - {word_count} words")
                
                if retry_count < max_retries - 1:
                    logger.info(f"Retrying Act {act_num}, Section {section_num}...")
                    return self.generate_section(act_num, section_num, section_data, retry_count + 1, max_retries)
                else:
                    logger.error(f"  ✗ Act {act_num}, Section {section_num} failed after {max_retries} attempts")
                    return f"[Error: Failed to generate Act {act_num}, Section {section_num} after retries]"
                    
        except Exception as e:
            logger.error(f"  ✗ Generation error: {e}")
            return f"[Error: Could not generate Act {act_num}, Section {section_num}]"

    def _clean_text(self, text: str) -> str:
        # Handle encoding issues
        try:
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        
        text = text.strip()
        lines = text.split('\n')
        
        # Remove unwanted instructional content
        unwanted_patterns = re.compile(
            r'^(WRITING STYLE|INSTRUCTIONS|SECTION|You are|STORY TITLE|CURRENT SECTION|SECTION GOAL|'
            r'PREVIOUSLY IN THE STORY|CHARACTER CONSISTENCY|CRITICAL INSTRUCTIONS)|'
            r'^(Please critique this section)|^---$|^\*\*\*$', 
            re.IGNORECASE
        )
        
        cleaned_lines = [line for line in lines if not unwanted_patterns.match(line.strip())]
        return '\n\n'.join(cleaned_lines).strip()

    def _validate_section(self, section_text: str, target_words: int) -> bool:
        if not section_text or section_text.startswith("[Error"):
            return False
            
        word_count = len(section_text.split())
        min_words = target_words * self.health_threshold
        
        if word_count < min_words:
            logger.warning(f"    - Validation failed: {word_count} words below minimum {int(min_words)}")
            return False
            
        return True

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        if not self.model_loaded:
            raise ConnectionError("Model not loaded.")
            
        logger.info(f"\n=== Generating Story: '{story_idea['title']}' ===")
        
        # Initialize story state
        self.current_story_data = story_idea
        self.section_summaries = []
        self.previous_section_text = ""
        
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                 story_idea['title'], "Starting story generation", 
                                 "6 acts, 2 sections each, 600 words per section")
        
        story_structure = self.create_story_structure(story_idea)
        full_story_sections = []
        total_words = 0
        
        for act in story_structure:
            act_num = act['act']
            act_title = act['title']
            logger.info(f"\n--- Starting Act {act_num}: {act_title} ---")
            
            # Add act separator (except for first act)
            if act_num > 1:
                full_story_sections.append(f"\n\n* * *\n\n")
            
            for section in act['sections']:
                section_num = section['section']
                section_text = self.generate_section(act_num, section_num, section)
                
                if not section_text.startswith("[Error"):
                    full_story_sections.append(f"{section_text}\n\n")
                    total_words += len(section_text.split())
                else:
                    full_story_sections.append(f"_{section_text}_\n\n")
                
                # Clean up memory
                gc.collect()
        
        complete_story_text = "".join(full_story_sections)
        
        logger.info("\n=== Story Generation Complete ===")
        logger.info(f"Total words generated: {total_words}")
        
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                 story_idea['title'], "✅ Story generation completed!", 
                                 f"Total: {total_words} words")
        
        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words
        }

def clean_for_audio(text: str) -> str:
    """Clean text for audio generation by removing formatting and extraneous content."""
    logger.info("Cleaning text for audio narration...")
    
    # Remove markdown headers
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markdown
    text = text.replace('**', '').replace('*', '')
    
    # Remove section separators
    text = re.sub(r'^\s*---\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\s*\*\s*\*\s*$', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    logger.info("Text cleaning for audio complete.")
    return text

def generate_audio(text: str, output_file: str = 'outputs/complete_narration.wav'):
    try:
        logger.info("Generating audio narration...")
        if telegram_handler:
            send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                 "Audio Generation", "Starting audio narration", 
                                 "Converting text to speech")
        
        os.makedirs('outputs', exist_ok=True)
        pipeline = KPipeline(lang_code='a')
        audio_segments = []
        
        # Split text into manageable chunks
        text_chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]
        total_chunks = len(text_chunks)
        logger.info(f"Processing {total_chunks} text chunks for audio generation")
        
        for i, chunk in enumerate(text_chunks):
            if chunk:
                logger.info(f"Processing audio chunk {i+1}/{total_chunks}")
                if telegram_handler and i % 10 == 0:
                    progress = f"Processing chunk {i+1}/{total_chunks}"
                    percentage = (i/total_chunks)*100
                    send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                         "Audio Generation", progress, 
                                         f"{percentage:.1f}% complete")
                
                generator = pipeline(chunk, voice='bm_fable')
                for _, _, audio in generator:
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
        
        if audio_segments:
            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            logger.info(f"Audio saved as '{output_file}' - Duration: {duration:.2f} seconds")
            
            if telegram_handler:
                send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                     "Audio Generation", "✅ Audio generation completed!", 
                                     f"Duration: {duration:.2f} seconds")
            return output_file
        else:
            logger.warning("No audio segments were generated")
            return None
            
    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        return None

def main():
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
        logger.info(f"Story idea extracted: '{story_idea.get('title', 'N/A')}'")
        
        generator = SuburbanHorrorGenerator()
        logger.info("Generating story...")
        final_story = generator.generate_complete_story(story_idea)
        
        if final_story and final_story['word_count'] > 0:
            story_title = final_story['title'].replace(' ', '_')
            
            # Save story with metadata
            story_file_md = output_dir / f"{story_title}.md"
            with open(story_file_md, 'w', encoding='utf-8') as f:
                f.write(f"# {final_story['title']}\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{final_story['text']}")
            logger.info(f"Story saved to {story_file_md}")
            
            # Clean and save text for audio (without word count references)
            audio_text = clean_for_audio(final_story['text'])
            clean_text_file = output_dir / f"{story_title}_Clean.txt"
            with open(clean_text_file, 'w', encoding='utf-8') as f:
                f.write(audio_text)
            logger.info(f"Clean audio text saved to {clean_text_file}")
            
            # Generate audio
            audio_file_path = str(output_dir / f"{story_title}_Audiobook.wav")
            audio_file = generate_audio(audio_text, audio_file_path)
            
            if audio_file and telegram_handler:
                logger.info(f"Audio file generated: {audio_file}")
                send_audio_file_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                          audio_file, final_story['title'])
            
            if telegram_handler:
                send_telegram_progress(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                     final_story['title'], "🎉 All tasks completed!", f"Story: {final_story['word_count']} words, Audio generated")
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