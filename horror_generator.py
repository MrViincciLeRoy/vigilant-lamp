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
        """
        Create a detailed story structure with scenes and plot points based on the story outline.
        This version remains hardcoded for consistency and predictability in CI/CD,
        but the content within scenes will be driven by the overall story_data.
        """
        return [
            {
                "scene": 1, "title": "The First Stain",
                "description": "Introduce Detective Naledi as she arrives at a horrifying crime scene in a seemingly peaceful Pretoria suburb.",
                "target_words": 400, "key_elements": ["Detective Naledi introduction", "Brutal crime scene details (implied, not explicit gore)", "Initial shock and questions", "Setting the grim tone in Pretoria"]
            },
            {
                "scene": 2, "title": "Whispers from the Past",
                "description": "Naledi uncovers a cold case from years ago with unsettling similarities, suggesting a serial pattern.",
                "target_words": 400, "key_elements": ["Discovery of old files/connection", "Similar modus operandi", "Initial dismissal by colleagues", "Naledi's growing unease"]
            },
            {
                "scene": 3, "title": "A Trail of Breadcrumbs",
                "description": "The killer leaves cryptic clues, drawing Naledi deeper into their twisted game.",
                "target_words": 400, "key_elements": ["Cryptic messages or symbols at crime scenes", "Killer taunting the police/Naledi directly", "Naledi follows a lead that takes her to a dark place", "Rising tension"]
            },
            {
                "scene": 4, "title": "The Shadow Falls",
                "description": "The killer escalates, targeting someone close to Naledi or leaving a particularly disturbing message.",
                "target_words": 400, "key_elements": ["Personal threat or near-miss for Naledi", "Heightened sense of danger", "A chilling realization about the killer's motive", "Isolation of Naledi"]
            },
            {
                "scene": 5, "title": "Beneath the Surface",
                "description": "Naledi discovers a horrific secret about Pretoria's dark underbelly connected to the killer's motives.",
                "target_words": 400, "key_elements": ["Uncovering a hidden network or dark history", "Motive is more twisted than initially thought", "Moral compromises or shocking revelations", "Naledi's determination amidst despair"]
            },
            {
                "scene": 6, "title": "The Unveiling",
                "description": "The climactic confrontation between Naledi and the killer, revealing their identity and the full horror.",
                "target_words": 400, "key_elements": ["Climactic showdown at a desolate location (e.g., abandoned building)", "Killer's identity exposed", "Twisted philosophy or justification", "Brutal physical and psychological struggle"]
            },
            {
                "scene": 7, "title": "Lingering Scars",
                "description": "The aftermath, exploring the psychological toll on Naledi and the lasting impact on Pretoria.",
                "target_words": 400, "key_elements": ["Resolution of the main conflict (capture or death of killer)", "Naledi's trauma and changed outlook", "Pretoria grappling with the unearthed horrors", "A somber, unsettling peace, not a happy ending"]
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

        # MODIFICATION: Focus on a concise summary of the last completed scene for continuity,
        # instead of the entire (potentially repetitive) history.
        # This prevents the model from getting stuck in loops or repeating old info.
        recent_context_summary = ""
        if previous_context:
            # Extract the last paragraph or two as a summary of the previous scene's end.
            # This is a heuristic; a more sophisticated summary could be used.
            last_paragraphs = previous_context.strip().split('\n\n')[-2:]
            recent_context_summary = "\n\n".join(last_paragraphs)

        # MODIFICATION: Stronger, more explicit prompt to ensure progression.
        if scene_num == 1:
            prompt = f"""You are a master of brutal, realistic horror, in the style of Jack Ketchum. Write the gripping opening scene of a true-crime story set in Pretoria, South Africa, focusing on the immediate shock and visceral impact of a horrifying, meticulously cruel crime. Avoid supernatural elements.
STORY TITLE: {story_data['title']}
STORY OUTLINE (for context, do not explicitly reference in text unless needed for plot): {story_outline}
SCENE 1: {title} - {description}
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}.
INSTRUCTIONS: Write around {target_words} words. Focus on introducing Detective Naledi, the grim reality of the crime scene, and setting a tone of unsettling dread. The horror should come from human depravity. Begin the story now.
"""
        else:
            prompt = f"""You are continuing a brutal, realistic horror story titled "{story_data['title']}" set in Pretoria.
LAST SCENE'S END (for continuity, do not repeat or re-introduce what's already known):
...{recent_context_summary}

CURRENT TASK: Write Scene {scene_num}: "{title}" - {description}
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}.
INSTRUCTIONS: Write the next part of the story, about {target_words} words. **Ensure the narrative progresses logically and ruthlessly from the previous scene, focusing on the new key elements. Do not repeat previous details or re-introduce characters unless vital for new plot points. Advance the plot significantly, maintaining a grim, unflinching, and suspenseful tone.** Continue seamlessly:
"""

        print(f"Generating Scene {scene_num}: '{title}' (target: {target_words} words)...")
        # MODIFICATION: Reduced max_tokens to be closer to the target to prevent excessive generation,
        # and slightly lowered temperature for more focused output.
        max_tokens = int(target_words * 1.3) # More conservative multiplier
        temperature = 0.65 # Slightly lower for less randomness, more focus

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["###", "---", "The End", "Chapter", "CHAPTER", "\n\n\n", story_data['title']], # Added story title as stop token
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
        # MODIFICATION: More aggressive cleaning of unwanted phrases that models sometimes output
        text = re.sub(r'^(SCENE \d+:.*?|INSTRUCTIONS:.*?|KEY ELEMENTS TO INCLUDE:.*?|LAST SCENE\'S END:.*?|\.+)$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.lower().startswith(('scene', '[', 'instruction', 'key elements', 'last scene\'s end', 'current task', 'story so far'))]
        formatted_text = '\n\n'.join(lines)
        return re.sub(r'\n{3,}', '\n\n', formatted_text).strip()

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story from an internal outline"""
        if not self.model_loaded: raise ValueError("Model not loaded")

        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")

        # The structure is now based on the new Ketchum-style scenes
        story_structure = self.create_story_structure(story_data)
        full_story_text = ""
        story_scenes, total_words, failed_scenes = [], 0, 0

        outline_for_prompt = story_data.get('outline', '')

        for scene_data in story_structure:
            # Pass the current full story text as context
            scene_text = self.generate_scene(scene_data, outline_for_prompt, full_story_text)

            if scene_text and not scene_text.startswith("[Error"):
                story_scenes.append({'title': scene_data['title'], 'text': scene_text})
                # MODIFICATION: Only add actual generated text to context
                full_story_text += "\n\n" + scene_text
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

        # Catch StopIteration if audio_generator is empty
        try:
            combined_audio = np.concatenate(list(audio_generator), axis=0)
        except ValueError: # Catch ValueError for empty sequence
            print("No audio segments were successfully generated (empty audio_generator).")
            return None

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

def main(story_outline_str: str):
    """Main execution function, now accepting story outline as an argument."""
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
    # --- New Jack Ketchum-style Story Prompt for Pretoria Crimes ---
    jack_ketchum_story_outline = """
# The Jacaranda Butcher
## Story Outline
Pretoria, present day. The city's deceptive tranquility is shattered by a series of meticulously cruel and grotesque murders, targeting individuals seemingly unconnected – a prominent architect, a respected high school teacher, and a struggling street artist. The killer leaves no forensic evidence, only a signature: each victim is found horrifically disfigured, with a single jacaranda blossom placed deliberately near their desecrated body. Detective Naledi Mohapi, a seasoned but weary investigator, is assigned the case. She quickly realizes these aren't random acts of violence; there's a cold, calculated motive, a deep-seated grievance. As Naledi delves into the victims' pasts, she uncovers a dark, shared secret from decades ago, a forgotten atrocity linked to a specific institution or event in Pretoria's history. The killer, dubbed "The Jacaranda Butcher" by the horrified media, is meticulously executing a long-dormant vengeance, and Naledi soon finds herself caught in a horrifying game of cat and mouse, where every clue is a taunt, and every step closer puts her in mortal peril. The horror is in the chilling human capacity for sustained malice and the hidden depravity beneath a beautiful city's surface.

## Key Characters
- **Detective Naledi Mohapi:** Mid-40s, sharp, jaded, and haunted by past cases. Driven to understand the 'why' behind the cruelty.
- **The Jacaranda Butcher:** The methodical, brutal killer. Their identity is unknown initially, but they are driven by a profound, long-held grudge stemming from a past injustice.
- **Dr. Elias Thorne:** A retired historian and former colleague of one of the victims, who holds a piece of the puzzle but is terrified to speak.

## Setting
- **Pretoria, South Africa:** From the leafy, affluent suburbs of Waterkloof and Brooklyn (where victims are found), to the grimy, forgotten industrial areas, and the historically rich but often unsettling governmental buildings and archives. The jacaranda trees, usually a symbol of beauty, become an ominous motif.
- **Time Period:** Present day, with flashbacks and discoveries revealing buried atrocities from the late apartheid era or immediate post-apartheid period.
"""

    # Call the main function with the new story outline
    main(jack_ketchum_story_outline)