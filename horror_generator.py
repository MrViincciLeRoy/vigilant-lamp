# -*- coding: utf-8 -*-
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import Dict, List, Optional
import gc
import kokoro
import soundfile as sf
import numpy as np
import requests
import re
import json

# --- MODEL LOADING (Optimized for CPU) ---
try:
    print("Loading Llama model for story generation...")
    # This setup is optimized for CPU-only environments.
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=8192,         # Context window size
        n_batch=512,        # Batch size for processing
        n_threads=None,     # Use all available CPU cores
        n_gpu_layers=0,     # Explicitly disable GPU layers
        verbose=False
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the language model. Error: {e}")
    llm = None

# --- STORY GENERATOR CLASS ---
class KetchumStyleHorrorGenerator:
    """
    Generates complete, 10-chapter horror stories in the style of Jack Ketchum,
    set against the backdrop of real-life South African crime.
    """
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.scene_summaries = []
        self.current_story_data = {}
        self.total_target_words = 4000  # Target for a 10-chapter story
        self.health_threshold = 0.75    # Minimum acceptable word count ratio for a scene

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Dynamically generates a 10-chapter story structure based on a high-level idea.
        This ensures a consistent narrative arc while adapting to any story.
        """
        # Extract key entities from the user's initial idea.
        protagonist = story_idea.get('protagonist', 'a determined detective')
        antagonist = story_idea.get('antagonist', 'a brutal killer')
        setting = story_idea.get('setting', 'Pretoria, South Africa')
        crime = story_idea.get('crime', 'a series of gruesome murders')
        signature = story_idea.get('signature', 'a unique, disturbing calling card')

        # This 10-chapter structure is a template that follows a classic three-act narrative arc.
        # It is now filled dynamically with the user's specific story elements.
        return [
            # ACT 1: The Setup
            {"scene": 1, "title": "The Inciting Incident", "description": f"The story opens with the discovery of a shocking crime related to '{crime}'. We meet {protagonist} as they are drawn into the case, immediately establishing the grim tone and the killer's brutal methods.", "target_words": 400, "focus": "character_introduction"},
            {"scene": 2, "title": "The Investigation Begins", "description": f"As {protagonist} starts investigating, the scale of the horror becomes clear. A second victim or discovery confirms a serial pattern, introducing the killer's '{signature}' and mounting pressure from media and superiors in {setting}.", "target_words": 400, "focus": "investigation_begins"},
            {"scene": 3, "title": "A Glimmer of a Lead", "description": f"The investigation uncovers the first tangible connection between the victims--a shared secret, a past event, or a common affiliation. This clue opens up the first real avenue of investigation but also hints at a deeper, more complex motive.", "target_words": 400, "focus": "clues_and_escalation"},

            # ACT 2: The Confrontation
            {"scene": 4, "title": "The First Obstacle", "description": "A promising lead turns into a dead end or a red herring, wasting precious time and increasing the protagonist's frustration. This obstacle highlights the challenges of the case and the cunning nature of the antagonist.", "target_words": 400, "focus": "misdirection"},
            {"scene": 5, "title": "The Turning Point", "description": f"Digging deeper, {protagonist} uncovers a dark, buried secret about the {setting} or the victims' past. This revelation fundamentally shifts the understanding of the killer's motive from simple brutality to something more personal and twisted.", "target_words": 400, "focus": "revelation"},
            {"scene": 6, "title": "The Stakes Become Personal", "description": f"The {antagonist}, now aware of the investigation closing in, makes a bold move. They leave a direct message for {protagonist} or target someone close to them, making the conflict deeply personal and dangerous.", "target_words": 400, "focus": "personal_stakes"},
            {"scene": 7, "title": "The Breakthrough", "description": f"Following the personal threat, {protagonist} connects with a key witness or finds the final piece of evidence that illuminates the killer's identity and endgame. This is the crucial 'aha' moment where the full, horrifying truth begins to emerge.", "target_words": 400, "focus": "breakthrough"},

            # ACT 3: The Resolution
            {"scene": 8, "title": "Dark Night of the Soul", "description": f"Armed with the truth but facing institutional roadblocks or their own fear, {protagonist} hits a low point. They must gather their resolve for the final confrontation, understanding the immense risk involved.", "target_words": 400, "focus": "character_depth"},
            {"scene": 9, "title": "The Climax", "description": f"Following the clues to a meaningful location, {protagonist} finally confronts the {antagonist}. This is a brutal, visceral, and psychologically charged showdown where the killer's full depravity and motivations are laid bare.", "target_words": 400, "focus": "climax"},
            {"scene": 10, "title": "The Aftermath", "description": f"The case is closed, but the horror leaves an indelible scar on {protagonist} and the community of {setting}. The ending is conclusive but somber, reflecting on the psychological toll and the nature of the evil that was faced. Justice is served, but innocence is lost.", "target_words": 400, "focus": "resolution"}
        ]

    def _build_focused_context(self, scene_num: int) -> str:
        """Builds a concise summary of previous scenes to maintain context."""
        if not self.scene_summaries:
            return ""
        # For later scenes, focus on the most recent events to keep the context sharp.
        relevant_summaries = self.scene_summaries[-2:] if scene_num > 2 else self.scene_summaries
        return "PREVIOUSLY IN THE STORY:\n" + "\n".join(relevant_summaries)

    def generate_scene(self, scene_data: Dict[str, str]) -> str:
        """Generates a single scene using a highly detailed, style-specific prompt."""
        if not self.model_loaded:
            return "[Error: Model not loaded]"

        scene_num = scene_data["scene"]
        title = scene_data["title"]
        description = scene_data["description"]
        target_words = scene_data["target_words"]

        # This is the core of the style enforcement.
        # This prompt is designed to force the LLM to adhere to the Jack Ketchum style.
        ketchum_style_guide = (
            "WRITING STYLE: Emulate Jack Ketchum. This means:\n"
            "- **Visceral Realism:** Focus on the physical and psychological reality of violence. Make it feel real and harrowing, not stylized or 'cool'.\n"
            "- **Human-Centered Horror:** The horror must come from human depravity, not supernatural elements. Explore the darkness within ordinary people.\n"
            "- **Psychological Depth:** Show the characters' internal thoughts, fears, and breaking points. The 'why' behind the violence is as important as the 'what'.\n"
            "- **Unflinching Pace:** Do not shy away from the brutal details, but keep the narrative moving forward. Build tension through steady, relentless progression.\n"
            "- **Authentic Setting:** Ground the story in the realities of modern Pretoria. Reference real locations, social tensions, and the atmosphere of the city to make the horror more immediate."
        )

        context = self._build_focused_context(scene_num)

        prompt = f"""
You are a master horror writer, channeling the spirit of Jack Ketchum.
Your task is to write a chapter for a brutal, realistic horror story set in Pretoria, South Africa.

STORY TITLE: "{self.current_story_data['title']}"
{context}

CURRENT CHAPTER: Chapter {scene_num} - {title}
CHAPTER GOAL: {description}

{ketchum_style_guide}

INSTRUCTIONS:
- Write approximately {target_words} words for this chapter.
- Advance the plot based on the 'CHAPTER GOAL'. Do NOT repeat past events.
- Focus on NEW actions, discoveries, and character developments.
- Do NOT use chapter headers, titles, or any meta-commentary in your response.
- Begin writing the chapter now.
"""
        print(f"Generating Chapter {scene_num}: '{title}'...")
        try:
            output = self.llm(
                prompt,
                max_tokens=int(target_words * 1.2),
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.15,
                stop=["\n\n\n", "Chapter", "CHAPTER", "The End"],
                echo=False
            )
            scene_text = self._clean_text(output['choices'][0]['text'])

            if self._validate_scene(scene_text, target_words):
                word_count = len(scene_text.split())
                print(f"  ✓ Chapter {scene_num} completed ({word_count} words).")
                # Create and store a summary for the next scene's context
                summary = f"In Chapter {scene_num}, {description}"
                self.scene_summaries.append(summary)
                return scene_text
            else:
                print(f"  ✗ Chapter {scene_num} failed quality check. Discarding.")
                return "[Error: Scene generation failed quality validation]"

        except Exception as e:
            print(f"  ✗ Error during scene generation: {e}")
            return f"[Error: Could not generate Chapter {scene_num}]"

    def _clean_text(self, text: str) -> str:
        """A simple utility to clean up the model's output."""
        text = text.strip()
        # Remove any lingering prompt instructions from the output
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.match(r'^(WRITING STYLE|INSTRUCTIONS|CHAPTER|You are|STORY TITLE)', line, re.IGNORECASE)]
        return '\n\n'.join(cleaned_lines).strip()

    def _validate_scene(self, scene_text: str, target_words: int) -> bool:
        """Validates that the generated scene meets basic quality criteria."""
        if not scene_text or scene_text.startswith("[Error"):
            return False
        word_count = len(scene_text.split())
        # Check if the scene is reasonably close to the target word count
        if word_count < target_words * self.health_threshold:
            print(f"    - Validation failed: Word count ({word_count}) is below threshold.")
            return False
        return True

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        """Orchestrates the generation of a full 10-chapter story."""
        if not self.model_loaded:
            raise ConnectionError("Cannot generate story because the language model is not loaded.")

        print(f"\n=== Generating New Horror Story: '{story_idea['title']}' ===")
        self.current_story_data = story_idea
        self.scene_summaries = [] # Reset for new story

        story_structure = self.create_story_structure(story_idea)
        full_story_chapters = []
        total_words = 0

        for scene_data in story_structure:
            chapter_text = self.generate_scene(scene_data)
            if not chapter_text.startswith("[Error"):
                full_story_chapters.append(f"## Chapter {scene_data['scene']}: {scene_data['title']}\n\n{chapter_text}")
                total_words += len(chapter_text.split())
            else:
                # If a chapter fails, we add a placeholder to indicate the failure.
                full_story_chapters.append(f"## Chapter {scene_data['scene']}: {scene_data['title']}\n\n_{chapter_text}_")
            gc.collect() # Free up memory after each scene

        complete_story_text = "\n\n---\n\n".join(full_story_chapters)

        print("\n=== Story Generation Complete ===")
        print(f"Final Word Count: {total_words} / {self.total_target_words}")

        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words,
        }

# --- UTILITY FUNCTIONS ---
def extract_story_idea_from_prompt(prompt_text: str) -> Dict[str, str]:
    """Parses a markdown-style prompt to extract key story elements."""
    idea = {'title': 'Untitled Horror Story'}
    lines = prompt_text.split('\n')
    for line in lines:
        if line.startswith('# '):
            idea['title'] = line[2:].strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            # Clean up key to be a valid identifier
            key = key.strip().lower().replace(' ', '_').replace('-', '_')
            idea[key] = value.strip()
    return idea

def save_story(story: Dict[str, any], output_dir: str = 'outputs') -> None:
    """Saves the generated story to a Markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r'[^\w\s-]', '', story['title']).strip().replace(' ', '_')
    filename = os.path.join(output_dir, f"{safe_title}.md")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {story['title']}\n\n")
        f.write(f"**Word Count:** {story['word_count']}\n\n")
        f.write(story['text'])
    print(f"Story successfully saved to: {filename}")

def generate_and_save_audiobook(story: Dict[str, any], output_dir: str = 'outputs') -> None:
    """
    Converts the story text to a WAV audio file and saves it.
    This function assumes the 'kokoro' library is installed and operational.
    It processes the story chapter by chapter to manage memory usage.
    """
    print("\n--- Starting Audiobook Generation ---")
    try:
        # Based on research, Kokoro TTS can be initialized and used this way. [7, 9]
        # This assumes you have the necessary model files ('kokoro-v0_19.onnx', 'voices.json')
        # in the same directory or accessible path.
        print("  - Initializing Kokoro TTS engine...")
        tts_engine = kokoro.Kokoro("kokoro-v0_19.onnx", "voices.json")
        print("  - TTS engine initialized.")

        # Prepare the text for speech synthesis
        full_text = story['text']
        # 1. Split the story into chapters for individual processing.
        # This is more robust than splitting by paragraphs alone.
        chapters = re.split(r'## Chapter \d+: .*?\n\n', full_text)
        chapters = [ch.strip() for ch in chapters if ch.strip() and not ch.startswith('[Error')]

        if not chapters:
            print("  ✗ No valid text content found to generate audio.")
            return

        print(f"  - Synthesizing {len(chapters)} chapters...")

        audio_segments = []
        sample_rate = 24000  # Kokoro TTS typically uses a 24000Hz sample rate

        # Process each chapter and collect the audio data
        for i, chunk in enumerate(chapters):
            print(f"    - Processing chapter {i+1}/{len(chapters)}...")
            # Clean up any remaining markdown for smoother speech
            clean_chunk = chunk.replace('\n\n---\n\n', '\n\n')
            # The create method returns a NumPy array of the audio waveform. [9]
            wav = tts_engine.create(clean_chunk)
            audio_segments.append(wav)
            gc.collect()

        # Concatenate all audio segments into one array
        print("  - Concatenating audio segments...")
        full_audio = np.concatenate(audio_segments)

        # Save the final audio file using soundfile. [3, 10]
        safe_title = re.sub(r'[^\w\s-]', '', story['title']).strip().replace(' ', '_')
        filename = os.path.join(output_dir, f"{safe_title}_Audiobook.wav")

        print(f"  - Saving audiobook to: {filename}")
        sf.write(filename, full_audio, sample_rate)
        print("  ✓ Audiobook generated and saved successfully.")

    except NameError:
        print("\n--- Audiobook Generation Skipped ---")
        print("  ✗ The 'kokoro' library is not available in your environment.")
        print("  - Please ensure it is installed and configured to enable this feature.")
    except FileNotFoundError:
        print("\n--- Audiobook Generation Failed ---")
        print("  ✗ Could not find Kokoro TTS model files (e.g., 'kokoro-v0_19.onnx').")
        print("  - Please make sure the required model files are in the script's directory.")
    except Exception as e:
        print(f"\n  ✗ An unexpected error occurred during audiobook generation: {e}")
        import traceback
        traceback.print_exc()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- STORY PROMPT ---
    # Define your high-level story idea here.
    # The script will use these details to build a full 10-chapter narrative.
    story_prompt = """
# Jacaranda Tears
Protagonist: Detective Lerato Phiri, a sharp but haunted investigator.
Antagonist: A killer known as 'The Gardener', who targets corrupt officials.
Setting: The affluent suburbs and decaying inner-city of Pretoria.
Crime: A series of ritualistic murders of high-profile figures.
Signature: A single, pressed jacaranda blossom placed on the victim's eyes.
"""

    try:
        # 1. Initialize the generator
        generator = KetchumStyleHorrorGenerator()

        # 2. Extract the core idea from the prompt
        story_idea = extract_story_idea_from_prompt(story_prompt)

        # 3. Generate the full story
        final_story = generator.generate_complete_story(story_idea)

        # 4. Save the result
        if final_story and final_story['word_count'] > 0:
            save_story(final_story)
            # 5. Generate and save the audiobook from the final story
            generate_and_save_audiobook(final_story)
        else:
            print("Story generation resulted in empty content. Nothing to save.")

    except Exception as e:
        print(f"\nAn unexpected fatal error occurred: {e}")
        import traceback
        traceback.print_exc()