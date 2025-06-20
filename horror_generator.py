# -*- coding: utf-8 -*-
"""Generic Horror Story Generator - Cohesive 10k Word Story with Full Context Usage"""
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

# Initialize the model with optimized settings
try:
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=8192,  # Use full context window
        n_batch=512,  # Larger batch size for better performance
        n_threads=None,  # Use all available threads
        verbose=False
    )
    print("Model loaded successfully with full context window (8192 tokens)")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

class GenericHorrorGenerator:
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.story_context = ""
        self.character_profiles = {}
        self.plot_points = []
        self.current_scene = 1
        self.total_target_words = 10000
        self.health_threshold = 0.8  # 80% minimum word generation health
        
    def create_story_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Create a detailed story structure with scenes and plot points based on the story outline"""
        outline = story_data.get('outline', '')
        title = story_data.get('title', 'Untitled Horror Story')
        scenes = self._analyze_outline_for_scenes(outline, title)
        if len(scenes) < 5:
            scenes = self._create_generic_structure(story_data)
        return scenes
    
    def _analyze_outline_for_scenes(self, outline: str, title: str) -> List[Dict[str, str]]:
        """Analyze the outline to determine natural scene breaks"""
        scenes = []
        outline_lower = outline.lower()
        if any(word in outline_lower for word in ['murder', 'kill', 'death', 'ritual', 'crime', 'unsolved']):
            scenes = [
                {
                    "scene": 1, "title": "The Cold Case",
                    "description": "Introduce the protagonist and their fascination with a series of unsolved crimes from Pretoria's past.",
                    "target_words": 1400, "key_elements": ["Protagonist introduction", "Setting the scene in modern Pretoria", "Discovery of the old case"]
                },
                {
                    "scene": 2, "title": "First Dig",
                    "description": "The protagonist begins their investigation, accessing archives and finding the first hints of a cover-up.",
                    "target_words": 1400, "key_elements": ["Initial research", "Redacted files", "Sense of being watched"]
                },
                {
                    "scene": 3, "title": "A Warning",
                    "description": "The protagonist receives a direct or indirect threat, making them realize the danger is real and present.",
                    "target_words": 1400, "key_elements": ["A veiled threat", "Escalating paranoia", "Contact with a reluctant source"]
                },
                {
                    "scene": 4, "title": "The Point of No Return",
                    "description": "A crucial piece of evidence is uncovered, but it puts the protagonist directly in the killer's sights.",
                    "target_words": 1400, "key_elements": ["Major breakthrough", "Direct confrontation or narrow escape", "Moral dilemma"]
                },
                {
                    "scene": 5, "title": "The Hunter and The Hunted",
                    "description": "Paranoia turns to terror as the protagonist realizes they are being actively hunted by the shadowy figure from the past.",
                    "target_words": 1400, "key_elements": ["Intensifying pursuit", "Isolation of the protagonist", "Desperate measures for survival"]
                },
                {
                    "scene": 6, "title": "The Ghost of Pretoria",
                    "description": "The climax where the protagonist confronts the killer and the horrifying truth of the original crimes is revealed.",
                    "target_words": 1400, "key_elements": ["Climactic confrontation", "Killer's identity and motive revealed", "Life-or-death struggle"]
                },
                {
                    "scene": 7, "title": "Jacaranda Scars",
                    "description": "The aftermath, showing the psychological and physical scars left on the protagonist and the resolution of the case.",
                    "target_words": 1600, "key_elements": ["Resolution of the main conflict", "Lingering consequences", "A somber sense of justice"]
                }
            ]
        return scenes
    
    def _create_generic_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Create a generic horror story structure when outline analysis fails"""
        return [
            {"scene": 1, "title": "The Beginning", "description": "Establish characters, setting, and the world before horror intrudes.", "target_words": 1400, "key_elements": ["Character introduction", "Setting", "Normal world"]},
            {"scene": 2, "title": "First Shadows", "description": "Something dark enters the story, first hints of the horror to come.", "target_words": 1400, "key_elements": ["Inciting incident", "First supernatural/horror element", "Character reactions"]},
            {"scene": 3, "title": "Growing Darkness", "description": "The horror elements strengthen, characters begin to change.", "target_words": 1400, "key_elements": ["Horror escalation", "Character development", "Building tension"]},
            {"scene": 4, "title": "The Turning Point", "description": "A crucial moment that changes everything for the characters.", "target_words": 1400, "key_elements": ["Major plot twist", "Character revelations", "Point of no return"]},
            {"scene": 5, "title": "Into the Abyss", "description": "Characters descend deeper into horror, hope begins to fade.", "target_words": 1400, "key_elements": ["Desperation", "Horror intensifies", "Character struggles"]},
            {"scene": 6, "title": "The Heart of Darkness", "description": "The climactic horror scene, the darkest moment of the story.", "target_words": 1400, "key_elements": ["Climax", "Ultimate horror", "Character fates"]},
            {"scene": 7, "title": "What Remains", "description": "The aftermath, consequences, and any resolution or closure.", "target_words": 1600, "key_elements": ["Resolution", "Aftermath", "Final thoughts"]}
        ]

    def generate_scene(self, scene_data: Dict[str, str], story_structure: List[Dict], 
                      story_context: str, temperature: float = 0.75, top_p: float = 0.92) -> str:
        """Generate a single scene with full context awareness"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        scene_num = scene_data["scene"]
        title = scene_data["title"]
        description = scene_data["description"]
        target_words = scene_data["target_words"]
        key_elements = scene_data["key_elements"]
        context_length = min(6000, len(self.story_context))
        recent_context = self.story_context[-context_length:] if self.story_context else ""
        
        if scene_num == 1:
            prompt = f"""Write the opening scene of a psychological true-crime horror story set in Pretoria, South Africa.

STORY OUTLINE:
{story_context}

SCENE: {title}
DESCRIPTION: {description}
TARGET LENGTH: {target_words} words
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}

WRITING REQUIREMENTS:
- Write in third person with a rich character perspective.
- Use rich sensory details and atmospheric descriptions of Pretoria.
- Build psychological tension through character thoughts and observations.
- Show character motivations through actions.
- Maintain a dark, foreboding, and realistic tone.
- Write exactly {target_words} words.

Begin the story:"""
        else:
            prompt = f"""Continue this psychological true-crime horror story.

STORY OUTLINE (for reference):
{story_context}

PREVIOUS STORY CONTENT:
{recent_context}

CURRENT SCENE: {title}
SCENE DESCRIPTION: {description}
TARGET LENGTH: {target_words} words
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}

WRITING REQUIREMENTS:
- Maintain perfect narrative continuity.
- Keep character voices and personalities consistent.
- Escalate the psychological tension and sense of danger.
- Use rich sensory details and atmospheric descriptions.
- Show rather than tell.
- Write exactly {target_words} words.
- End with a natural transition point for the next scene.

Continue the story seamlessly:"""

        print(f"Generating Scene {scene_num}: '{title}' (target: {target_words} words)...")
        max_tokens = int(target_words * 1.8)
        best_attempt = ""
        best_word_count = 0
        
        for attempt in range(3):
            try:
                output = self.llm(
                    prompt, max_tokens=max_tokens, temperature=temperature + (attempt * 0.05),
                    top_p=top_p, repeat_penalty=1.05 + (attempt * 0.02),
                    stop=["###", "---", "THE END", "Chapter", "CHAPTER", "\n\n\n\n"], echo=False
                )
                scene_text = self._clean_and_format_text(output['choices'][0]['text'])
                word_count = len(scene_text.split())
                target_proximity = abs(word_count - target_words) / target_words
                best_proximity = abs(best_word_count - target_words) / target_words if best_word_count > 0 else 1.0
                
                if target_proximity < best_proximity and word_count >= (target_words * 0.7):
                    best_attempt = scene_text
                    best_word_count = word_count
                
                if word_count / target_words >= self.health_threshold:
                    best_attempt = scene_text
                    best_word_count = word_count
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        if best_attempt:
            health_ratio = best_word_count / target_words
            print(f"Scene {scene_num} completed: {best_word_count} words (Health: {health_ratio:.1%})")
            return best_attempt
        else:
            print(f"Failed to generate Scene {scene_num}")
            return f"[Error: Could not generate Scene {scene_num}: {title}]"

    def _clean_and_format_text(self, text: str) -> str:
        """Enhanced text cleaning and formatting"""
        if not text: return ""
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\*+|---+', '', text)
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('[') and not line.startswith('SCENE')]
        formatted_text = '\n\n'.join(lines)
        return re.sub(r'\n{3,}', '\n\n', formatted_text).strip()

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story from an internal outline"""
        if not self.model_loaded: raise ValueError("Model not loaded")
        
        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")
        
        story_structure = self.create_story_structure(story_data)
        story_scenes, total_words, failed_scenes = [], 0, 0
        
        for scene_data in story_structure:
            scene_text = self.generate_scene(scene_data, story_structure, story_data.get('outline', ''))
            if scene_text and not scene_text.startswith("[Error"):
                story_scenes.append({'title': scene_data['title'], 'text': scene_text})
                self.story_context += "\n\n" + scene_text
                scene_words = len(scene_text.split())
                total_words += scene_words
                health_ratio = scene_words / scene_data['target_words']
                print(f"✓ Scene {scene_data['scene']} added: {scene_words} words")
                print(f"  Progress: {total_words}/{self.total_target_words} words ({total_words/self.total_target_words*100:.1f}%)")
                print(f"  Scene Health: {health_ratio:.1%}")
                gc.collect()
            else:
                failed_scenes += 1
                print(f"✗ Scene {scene_data['scene']} failed")
        
        complete_story = "\n\n---\n\n".join([scene['text'] for scene in story_scenes])
        story_health = total_words / self.total_target_words if self.total_target_words > 0 else 0
        
        print("\n=== Story Generation Complete ===")
        print(f"Final word count: {total_words} words ({story_health:.1%} of target)")
        
        return {'title': story_data['title'], 'text': complete_story, 'word_count': total_words, 'target_words': self.total_target_words,
                'health_ratio': story_health, 'scenes_completed': len(story_scenes), 'scenes_failed': failed_scenes}

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
            original_title = story.get('original_title', story['title'])
            f.write(f"# {original_title}\n\n")
            f.write(story['text'])
            f.write("\n\n---\n\n## Story Statistics\n\n")
            f.write(f"- **Word Count:** {story['word_count']:,} words\n")
            f.write(f"- **Target:** {story['target_words']:,} words\n")
            f.write(f"- **Achievement:** {story['health_ratio']:.1%}\n")
            f.write(f"- **Scenes Completed:** {story['scenes_completed']}\n")
            f.write(f"- **Scenes Failed:** {story['scenes_failed']}\n")
            f.write(f"- **Health Status:** {'✓ HEALTHY' if story['health_ratio'] >= 0.8 else '⚠ BELOW THRESHOLD'}\n")
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
        
        audio_segments = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            try:
                # Process in smaller chunks to avoid overwhelming the TTS
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if sentence:
                        generator = pipeline(sentence, voice='bm_fable')
                        for _, _, audio in generator:
                            if audio is not None and len(audio) > 0:
                                audio_segments.append(audio)
            except Exception as e:
                print(f"Error processing audio for a paragraph segment: {e}")
            if (i + 1) % 10 == 0 or (i + 1) == total_paragraphs:
                print(f"Audio progress: {i + 1}/{total_paragraphs} paragraphs processed")

        if audio_segments:
            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            print(f"Audio saved: '{output_file}' (Duration: {duration/60:.2f} minutes)")
            return output_file
        else:
            print("No audio segments were generated.")
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

def print_final_report(story: Dict[str, str]):
    """Print a detailed final report"""
    print(f"\n{'='*50}\nFINAL STORY REPORT\n{'='*50}")
    original_title = story.get('original_title', story['title'])
    print(f"Title: {original_title}")
    print(f"Word Count: {story['word_count']:,} words")
    print(f"Achievement: {story['health_ratio']:.1%}")
    print(f"Health Status: {'✓ HEALTHY' if story['health_ratio'] >= 0.8 else '⚠ BELOW THRESHOLD'}")
    print(f"Scenes Completed: {story['scenes_completed']} / Failed: {story['scenes_failed']}")
    if story['health_ratio'] < 0.8:
        print("⚠ Story below health threshold - quality may be inconsistent.")
    print(f"{'='*50}")

def main():
    """Main execution function with a self-contained story outline."""
    
    # The story outline is now a multi-line string inside the script.
    # No external .md file is needed.
    story_outline_str = """
# The Waterkloof Silence

## Story Outline
In present-day Pretoria, Liam, a driven journalism student at Tuks, stumbles upon a microfiche archive detailing a series of unsolved murders from 1988. The victims, all young activists, were found along the tranquil Moreleta Spruit hiking trail. The case was dubbed "The Spruit Silencings" before it was abruptly classified and buried by the apartheid-era security apparatus.

Driven by a desire to uncover a hidden piece of Pretoria's history for his thesis, Liam begins to digitize the case files. He quickly finds that official reports are heavily redacted and key evidence is "missing." When he tries to contact the original investigating detective, now a recluse, he's met with a paranoid warning to "let the dead stay buried."

As Liam digs deeper, a chilling sense of being watched descends upon him. His flat in Hatfield is broken into, but nothing is stolen—only his research notes are disturbed. A shadowy car, an old Ford Sierra, seems to appear wherever he goes. The horror is not supernatural, but the terrifyingly real possibility that the powerful individual who silenced the activists forty years ago is still out there, and is now methodically silencing Liam's investigation.

## Key Characters
- **Liam van der Merwe:** 22-year-old journalism student at the University of Pretoria, idealistic and relentless.
- **The Man in the Sierra:** A shadowy, older figure who represents the unpunished evil of the past. His identity is unknown.
- **Retired Detective Bester:** The original investigator, now living in fear in a small North West town, tormented by the case he was forced to abandon.

## Setting
- **Pretoria, South Africa:** The leafy, jacaranda-lined suburbs like Waterkloof, the bustling student area of Hatfield, the sterile National Archives, and the ominously quiet Moreleta Spruit nature reserve.
- **Time Period:** Modern day, with journalistic research and flashbacks revealing the oppressive atmosphere of Pretoria in the late 1980s.
"""

    print("=== GENERIC HORROR STORY GENERATOR (Self-Contained Mode) ===")
    print("Target: 10,000 words | Health threshold: 80%")
    
    generator = GenericHorrorGenerator()
    if not generator.model_loaded:
        print("\nERROR: Model not loaded. Please check your setup.")
        return
    
    # Extract story data directly from the internal string
    story_data = extract_story_data(story_outline_str)
    
    # Generate the complete story
    story = generator.generate_complete_story(story_data)
    
    # Save the story and generate audio
    if story['word_count'] > 0:
        # Use the original title for metadata but a safe version for the filename
        story['original_title'] = story.get('title', 'story')
        
        story_file = save_story(story, 'outputs')
        
        if story_file:
            print_final_report(story)
            
            audio_file = None
            if story['health_ratio'] >= 0.8:
                print("\n=== AUDIO GENERATION ===")
                audio_file = generate_audio(story['text'], 'outputs', story['title'])
            else:
                print("\n=== SKIPPING AUDIO (Story health is below threshold) ===")
            
            print("\n=== TELEGRAM UPLOAD ===")
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            send_to_telegram(bot_token, chat_id, story_file, audio_file)
            
            print("\n=== PROCESS COMPLETE ===")
            print(f"Story saved to: {story_file}")
            if audio_file:
                print(f"Audio saved to: {audio_file}")
    else:
        print("\nERROR: Failed to generate or save the story.")

if __name__ == "__main__":
    main()