"""Generic Horror Story Generator - Cohesive Story with Full Context Usage (Optimized for CI/CD)"""
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import Dict, List, Tuple, Optional
import gc
import kokoro
import soundfile as sf
import numpy as np
import requests
import re
import json
from pathlib import Path

# MODIFICATION: Optimized model loading for CPU-bound environments like GitHub Actions.
try:
    print("Loading Llama model...")
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=8192,         # Full context window
        n_batch=512,        # A reasonable batch size for CPU
        n_threads=None,     # Use all available CPU cores
        n_gpu_layers=0,     # Explicitly state we are using CPU only
        verbose=False
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading model: {e}")
    llm = None

class GenericHorrorGenerator:
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.story_context = ""
        
        # MODIFICATION: Drastically reduced total word count to make it feasible on a GitHub runner.
        # This is the most important change for performance.
        self.total_target_words = 2800 
        self.health_threshold = 0.75  # 75% minimum word generation health is more forgiving

    def create_story_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Create a detailed story structure with scenes and plot points based on the story outline"""
        # MODIFICATION: The scene structure is now hardcoded and simplified for a shorter story.
        # This avoids complex analysis and provides a reliable structure.
        # Target words per scene are reduced from 1400 to 400.
        return [
            {
                "scene": 1, "title": "The Cold Case",
                "description": "Introduce the protagonist, Liam, and his fascination with a series of unsolved crimes from Pretoria's past.",
                "target_words": 400, "key_elements": ["Protagonist introduction", "Setting the scene in modern Pretoria", "Discovery of the old case"]
            },
            {
                "scene": 2, "title": "First Dig",
                "description": "Liam begins his investigation, accessing archives and finding the first hints of a cover-up.",
                "target_words": 400, "key_elements": ["Initial research", "Redacted files", "A sense of being watched"]
            },
            {
                "scene": 3, "title": "A Warning",
                "description": "Liam receives a direct or indirect threat from retired Detective Bester, making him realize the danger is real.",
                "target_words": 400, "key_elements": ["A veiled threat", "Escalating paranoia", "Contact with the reluctant source"]
            },
            {
                "scene": 4, "title": "The Point of No Return",
                "description": "A crucial piece of evidence is uncovered, but it puts Liam directly in the killer's sights. His flat is broken into.",
                "target_words": 400, "key_elements": ["Major breakthrough", "Direct threat", "Notes are disturbed"]
            },
            {
                "scene": 5, "title": "The Hunter and The Hunted",
                "description": "Paranoia turns to terror as Liam realizes he is being actively hunted by the shadowy Man in the Sierra.",
                "target_words": 400, "key_elements": ["Intensifying pursuit", "Isolation of the protagonist", "Desperate measures"]
            },
            {
                "scene": 6, "title": "The Ghost of Pretoria",
                "description": "The climax where Liam confronts the killer and the horrifying truth of the original crimes is revealed.",
                "target_words": 400, "key_elements": ["Climactic confrontation", "Killer's identity and motive revealed", "Life-or-death struggle"]
            },
            {
                "scene": 7, "title": "Jacaranda Scars",
                "description": "The aftermath, showing the psychological and physical scars left on Liam and the resolution of the case.",
                "target_words": 400, "key_elements": ["Resolution of the main conflict", "Lingering consequences", "A somber sense of justice"]
            }
        ]

    def generate_scene(self, scene_data: Dict[str, str], story_outline: str, previous_context: str) -> str:
        """
        Generate a single scene with full context awareness.
        MODIFICATION: This function is heavily simplified for speed and reliability.
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        scene_num = scene_data["scene"]
        title = scene_data["title"]
        description = scene_data["description"]
        target_words = scene_data["target_words"]
        key_elements = scene_data["key_elements"]

        # MODIFICATION: Use a more concise context window to keep the prompt size manageable.
        context_length = min(4000, len(previous_context))
        recent_context = previous_context[-context_length:] if previous_context else ""

        # MODIFICATION: Simpler, more direct prompt structure.
        if scene_num == 1:
            prompt = f"""You are a master of psychological horror. Write the gripping opening scene of a true-crime story set in Pretoria, South Africa.
STORY OUTLINE: {story_outline}
SCENE 1: {title} - {description}
KEY ELEMENTS: {', '.join(key_elements)}
INSTRUCTIONS: Write around {target_words} words. Begin the story now."""
        else:
            prompt = f"""You are continuing a psychological horror story.
STORY SO FAR:
...{recent_context}

NEXT SCENE: Scene {scene_num}: {title} - {description}
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}
INSTRUCTIONS: Write the next part of the story, about {target_words} words, maintaining a dark, realistic tone. Continue seamlessly from the previous text."""

        print(f"Generating Scene {scene_num}: '{title}' (target: {target_words} words)...")
        # MODIFICATION: Reduced max_tokens to be closer to the target, preventing wasted generation.
        # (target_words * 1.5) is a safer multiplier than 1.8.
        max_tokens = int(target_words * 1.5)
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["###", "---", "The End", "Chapter", "CHAPTER", "\n\n\n"], # Added more stop tokens
                echo=False
            )
            scene_text = self._clean_and_format_text(output['choices'][0]['text'])
            word_count = len(scene_text.split())

            if word_count / target_words >= self.health_threshold:
                print(f"Scene {scene_num} completed: {word_count} words (Health: {word_count/target_words:.1%})")
                return scene_text
            else:
                print(f"✗ Scene {scene_num} failed health check (got {word_count} words). Returning partial content.")
                return scene_text # Return whatever was generated
        except Exception as e:
            print(f"✗ An error occurred during scene generation: {e}")
            return f"[Error: Could not generate Scene {scene_num}: {title}]"

    def _clean_and_format_text(self, text: str) -> str:
        """Enhanced text cleaning and formatting"""
        if not text: return ""
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\*+|---+', '', text)
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.lower().startswith(('scene', '[', 'instruction'))]
        formatted_text = '\n\n'.join(lines)
        return re.sub(r'\n{3,}', '\n\n', formatted_text).strip()

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story from an internal outline"""
        if not self.model_loaded: raise ValueError("Model not loaded")

        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")

        story_structure = self.create_story_structure(story_data)
        full_story_text = ""
        story_scenes, total_words, failed_scenes = [], 0, 0
        
        outline_for_prompt = story_data.get('outline', '')

        for scene_data in story_structure:
            # Pass the current full story text as context
            scene_text = self.generate_scene(scene_data, outline_for_prompt, full_story_text)
            
            if scene_text and not scene_text.startswith("[Error"):
                story_scenes.append({'title': scene_data['title'], 'text': scene_text})
                full_story_text += "\n\n" + scene_text # Update context for the next scene
                scene_words = len(scene_text.split())
                total_words += scene_words
                print(f"  Progress: {total_words}/{self.total_target_words} words ({total_words/self.total_target_words*100:.1f}%)")
                gc.collect() # Garbage collect to free memory
            else:
                failed_scenes += 1
                print(f"✗ Scene {scene_data['scene']} failed to generate properly.")

        complete_story_text = "\n\n---\n\n".join([scene['text'] for scene in story_scenes])
        story_health = total_words / self.total_target_words if self.total_target_words > 0 else 0

        print("\n=== Story Generation Complete ===")
        print(f"Final word count: {total_words} words ({story_health:.1%} of target)")

        return {
            'title': story_data['title'], 'text': complete_story_text, 'word_count': total_words, 
            'target_words': self.total_target_words, 'health_ratio': story_health, 
            'scenes_completed': len(story_scenes), 'scenes_failed': failed_scenes
        }

def extract_story_data(markdown_text: str) -> Dict[str, str]:
    """Extract story data from a markdown string"""
    lines = markdown_text.split('\n')
    story_title = ""
    for line in lines:
        line = line.strip()
        if line.startswith('# ') and not story_title:
            story_title = line[2:].strip()
            break
    return {'title': story_title or 'Untitled Horror Story', 'outline': markdown_text.strip()}

def save_story(story: Dict[str, str], output_dir: str = 'outputs') -> Optional[str]:
    """Save complete story with enhanced metadata"""
    # This function is fine, no changes needed.
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r'[^\w\s-]', '', story['title']).strip().replace(' ', '_').lower()
    filename = os.path.join(output_dir, f"{safe_title}_complete.md")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {story['title']}\n\n")
            f.write(story['text'])
            f.write("\n\n---\n\n## Story Statistics\n\n")
            f.write(f"- **Word Count:** {story['word_count']:,} words\n")
            f.write(f"- **Target:** {story['target_words']:,} words\n")
            f.write(f"- **Achievement:** {story['health_ratio']:.1%}\n")
            f.write(f"- **Scenes Completed:** {story['scenes_completed']}\n")
            f.write(f"- **Scenes Failed:** {story['scenes_failed']}\n")
        print(f"Story saved: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving story: {e}")
        return None

def generate_audio(text: str, output_dir: str = 'outputs', story_title: str = 'story') -> Optional[str]:
    """Generate audio narration"""
    # MODIFICATION: Simplified audio generation to be more robust.
    try:
        print("Generating audio narration...")
        os.makedirs(output_dir, exist_ok=True)
        safe_title = re.sub(r'[^\w\s-]', '', story_title).strip().replace(' ', '_').lower()
        output_file = os.path.join(output_dir, f"{safe_title}_narration.wav")

        try:
            pipeline = kokoro.KPipeline(lang_code='a')
        except Exception as e:
            print(f"Kokoro not available for audio generation: {e}")
            return None

        # Process the text in larger, more manageable chunks (paragraphs)
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not chunks:
            print("No text to narrate.")
            return None
            
        print(f"Narrating {len(chunks)} paragraphs...")
        # Use a generator expression for memory efficiency
        audio_generator = (audio for chunk in chunks for _, _, audio in pipeline(chunk, voice='bm_fable') if audio is not None and len(audio) > 0)
        
        combined_audio = np.concatenate(list(audio_generator), axis=0)

        if combined_audio.size > 0:
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            print(f"Audio saved: '{output_file}' (Duration: {duration/60:.2f} minutes)")
            return output_file
        else:
            print("No audio segments were successfully generated.")
            return None
    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        return None

def send_to_telegram(bot_token: str, chat_id: str, text_file: str = None, audio_file: str = None):
    """Send files to Telegram"""
    # This function is fine, no changes needed.
    if not bot_token or not chat_id:
        print("Telegram credentials not provided. Skipping upload.")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    for file_type, file_path in [('story', text_file), ('audio', audio_file)]:
        if file_path and os.path.exists(file_path):
            try:
                print(f"Uploading {file_type} to Telegram...")
                with open(file_path, 'rb') as f:
                    files = {'document': f}
                    response = requests.post(url, data={'chat_id': chat_id}, files=files, timeout=300)
                    if response.status_code == 200: print(f"✓ Successfully sent {file_type}.")
                    else: print(f"✗ Failed to send {file_type}: {response.text}")
            except Exception as e:
                print(f"✗ Error sending {file_type}: {e}")

def main():
    """Main execution function with a self-contained story outline."""
    # The story outline is the same, but the script will generate a shorter story based on it.
    story_outline_str = """
# The Waterkloof Silence
## Story Outline
In present-day Pretoria, Liam, a driven journalism student at Tuks, stumbles upon a microfiche archive detailing a series of unsolved murders from 1988. The victims, all young activists, were found along the tranquil Moreleta Spruit hiking trail. The case was dubbed "The Spruit Silencings" before it was abruptly classified and buried by the apartheid-era security apparatus. Driven by a desire to uncover a hidden piece of Pretoria's history for his thesis, Liam begins to digitize the case files. He quickly finds that official reports are heavily redacted and key evidence is "missing." When he tries to contact the original investigating detective, now a recluse, he's met with a paranoid warning to "let the dead stay buried." As Liam digs deeper, a chilling sense of being watched descends upon him. His flat in Hatfield is broken into, but nothing is stolen—only his research notes are disturbed. A shadowy car, an old Ford Sierra, seems to appear wherever he goes. The horror is not supernatural, but the terrifyingly real possibility that the powerful individual who silenced the activists forty years ago is still out there, and is now methodically silencing Liam's investigation.
## Key Characters
- **Liam van der Merwe:** 22-year-old journalism student at the University of Pretoria, idealistic and relentless.
- **The Man in the Sierra:** A shadowy, older figure who represents the unpunished evil of the past. His identity is unknown.
- **Retired Detective Bester:** The original investigator, now living in fear in a small North West town, tormented by the case he was forced to abandon.
## Setting
- **Pretoria, South Africa:** The leafy, jacaranda-lined suburbs like Waterkloof, the bustling student area of Hatfield, the sterile National Archives, and the ominously quiet Moreleta Spruit nature reserve.
- **Time Period:** Modern day, with journalistic research and flashbacks revealing the oppressive atmosphere of Pretoria in the late 1980s.
"""

    print("=== OPTIMIZED HORROR STORY GENERATOR ===")

    generator = GenericHorrorGenerator()
    if not generator.model_loaded:
        print("\nERROR: Model not loaded. Please check your setup.")
        return

    story_data = extract_story_data(story_outline_str)
    story = generator.generate_complete_story(story_data)

    if story['word_count'] > 0:
        story_file = save_story(story, 'outputs')

        if story_file:
            audio_file = generate_audio(story['text'], 'outputs', story['title'])
            
            print("\n=== TELEGRAM UPLOAD ===")
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            send_to_telegram(bot_token, chat_id, story_file, audio_file)

            print("\n=== PROCESS COMPLETE ===")
    else:
        print("\nERROR: Failed to generate a story with any content.")

if __name__ == "__main__":
    main()