# -*- coding: utf-8 -*-
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
        self.scene_summaries = []
        self.current_story_data = {}

        # MODIFICATION: Increased total word count for a 10-scene structure.
        self.total_target_words = 4000
        self.health_threshold = 0.75

    def create_story_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create a detailed 10-scene story structure for a deeper narrative.
        """
        return [
            {
                "scene": 1, "title": "The First Blossom",
                "description": "Detective Naledi Mohapi is called to a gruesome crime scene in an affluent Pretoria suburb. The horror is precise and personal.",
                "target_words": 400,
                "key_elements": ["Introduce Detective Naledi", "Details of the meticulous, brutal murder", "The killer's signature is found (e.g., a jacaranda blossom)", "Establish the grim, realistic tone"],
                "focus": "character_introduction"
            },
            {
                "scene": 2, "title": "The Second Victim",
                "description": "Before Naledi can make headway, a second body is found. The victim is from a different social class, but the signature is the same, deepening the mystery.",
                "target_words": 400,
                "key_elements": ["Discovery of a second, contrasting victim", "Confirms the serial nature of the crimes", "Media frenzy begins, coining a name for the killer", "Pressure mounts on Naledi and her team"],
                "focus": "investigation_begins"
            },
            {
                "scene": 3, "title": "A Shared Past",
                "description": "Naledi's team finds the first tangible link between the victims: a connection to a controversial and recently approved urban development project.",
                "target_words": 400,
                "key_elements": ["Intensive detective work (reviewing financials, records)", "The link between victims is a specific project or company", "Introduces potential motives (greed, corruption)", "Naledi identifies a new pool of potential victims"],
                "focus": "clues_and_escalation"
            },
            {
                "scene": 4, "title": "The Red Herring",
                "description": "A promising lead points to a disgruntled activist known for protesting the development. Naledi pursues this, but it feels too simple.",
                "target_words": 400,
                "key_elements": ["Naledi interviews a seemingly obvious suspect", "The suspect has an alibi or is clearly not the killer", "This dead-end wastes time and increases Naledi's frustration", "The real killer may use this distraction to plan their next move"],
                "focus": "misdirection"
            },
            {
                "scene": 5, "title": "Whispers from the Rubble",
                "description": "Naledi digs deeper into the history of the development site itself, uncovering stories of displaced families and forgotten tragedies.",
                "target_words": 400,
                "key_elements": ["Naledi visits the development site", "Talks to locals or old-timers about the area's history", "Uncovers a dark secret or injustice related to the land", "The motive appears to be rooted in historical grievance, not just business"],
                "focus": "revelation"
            },
            {
                "scene": 6, "title": "The Butcher's Taunt",
                "description": "The killer, aware of Naledi's investigation, leaves a direct and chilling message for her at the third crime scene, making the case deeply personal.",
                "target_words": 400,
                "key_elements": ["A third, shocking murder occurs", "A clue or message is left specifically for Naledi", "The killer demonstrates knowledge of the police investigation", "Naledi feels hunted and isolated"],
                "focus": "personal_stakes"
            },
            {
                "scene": 7, "title": "The Historian's Fear",
                "description": "Naledi seeks out an expert, like a retired historian or journalist, who holds the key to the past tragedy but is terrified to speak.",
                "target_words": 400,
                "key_elements": ["Naledi identifies a key witness or expert", "The character is reluctant and fearful, hinting at a powerful conspiracy", "They provide the final piece of the historical puzzle", "Naledi understands the true depth of the killer's rage"],
                "focus": "breakthrough"
            },
            {
                "scene": 8, "title": "Dark Night of the Soul",
                "description": "Overwhelmed by the horror and frustrated by institutional roadblocks, Naledi hits her lowest point, questioning her ability to stop the killer.",
                "target_words": 400,
                "key_elements": ["Naledi faces pressure from her superiors to close the case", "She revisits the case files, feeling hopeless", "A moment of reflection on the human capacity for cruelty", "Her resolve is tested, but she finds a new determination"],
                "focus": "character_depth"
            },
            {
                "scene": 9, "title": "Beneath the Jacaranda",
                "description": "Putting all the pieces together, Naledi identifies the killer and their lair, leading to a brutal and intimate confrontation.",
                "target_words": 400,
                "key_elements": ["The final 'aha!' moment of realization", "Naledi tracks the killer to a meaningful location", "The killer's identity and twisted justification are fully revealed", "A visceral, psychological, and physical struggle ensues"],
                "focus": "climax"
            },
            {
                "scene": 10, "title": "The Lingering Stain",
                "description": "In the aftermath, the case is closed, but Naledi and the city are irrevocably scarred. The beautiful jacarandas now hold a dark memory.",
                "target_words": 400,
                "key_elements": ["Resolution of the main conflict", "Naledi reflects on the case's psychological toll", "The city grapples with the unearthed horrors", "A somber, unsettling peace; justice is served, but the scars remain"],
                "focus": "resolution"
            }
        ]

    def _create_scene_summary(self, scene_text: str, scene_data: Dict[str, str]) -> str:
        """Create a concise summary of a scene for context in future scenes."""
        sentences = [s.strip() for s in scene_text.split('.') if s.strip()]
        key_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
        summary = f"Scene {scene_data['scene']} ({scene_data['title']}): " + '. '.join(key_sentences)
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary

    def _build_focused_context(self, scene_num: int) -> str:
        """Build focused context from previous scene summaries."""
        if not self.scene_summaries:
            return ""
        if scene_num <= 3:
            relevant_summaries = self.scene_summaries
        else:
            relevant_summaries = self.scene_summaries[-2:]
        context = "PREVIOUS PLOT DEVELOPMENTS:\n" + "\n".join(relevant_summaries)
        return context

    def generate_scene(self, scene_data: Dict[str, str], story_outline: str, previous_context: str) -> str:
        """
        Generate a single scene with improved context awareness and repetition prevention.
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        scene_num = scene_data["scene"]
        title = scene_data["title"]
        description = scene_data["description"]
        target_words = scene_data["target_words"]
        key_elements = scene_data["key_elements"]
        focus = scene_data.get("focus", "general")

        focused_context = self._build_focused_context(scene_num)

        progression_instructions = {
            "character_introduction": "Focus on establishing character depth and the immediate situation. Avoid backstory dumps.",
            "investigation_begins": "Advance the investigation with new discoveries. Show detective work in action.",
            "clues_and_escalation": "Introduce new evidence and raise the stakes. Build tension through discoveries.",
            "misdirection": "Introduce a plausible but incorrect lead. Show the challenges of the investigation.",
            "revelation": "Reveal crucial information that changes the understanding of the motive. Focus on the impact of discovery.",
            "personal_stakes": "Make the threat personal and immediate. Show character vulnerability.",
            "breakthrough": "Provide the final piece of the puzzle through a key witness or discovery.",
            "character_depth": "Explore the protagonist's internal struggle and resolve.",
            "climax": "Bring all elements together in confrontation. Focus on action and revelation.",
            "resolution": "Show consequences and aftermath. Avoid tying up loose ends too neatly."
        }
        
        if scene_num == 1:
            prompt = f"""You are writing a brutal, realistic horror story in the style of Jack Ketchum. Write the opening scene with visceral impact and psychological depth.

STORY: {self.current_story_data['title']}
SCENE: {title} - {description}

MANDATORY ELEMENTS TO INTRODUCE:
{chr(10).join(f"• {element}" for element in key_elements)}

WRITING INSTRUCTIONS:
- Write approximately {target_words} words
- {progression_instructions[focus]}
- Focus on immediate sensory details and psychological impact
- Avoid supernatural elements - horror comes from human depravity
- Establish the dark tone immediately
- Do NOT include scene headers, titles, or meta-commentary

Begin the story now:"""

        else:
            prompt = f"""Continue the horror story "{self.current_story_data['title']}" with the next scene. Ensure narrative progression and avoid repetition.

{focused_context}

CURRENT SCENE: {title} - {description}

MANDATORY NEW ELEMENTS TO INTRODUCE:
{chr(10).join(f"• {element}" for element in key_elements)}

CRITICAL INSTRUCTIONS:
- Write approximately {target_words} words
- {progression_instructions[focus]}
- Build directly on previous developments WITHOUT repeating them
- Introduce the mandatory elements as new plot developments
- Advance the story significantly - do not rehash previous scenes
- Maintain the dark, unflinching tone established earlier
- Do NOT include scene headers, titles, or meta-commentary
- Focus on NEW actions, discoveries, and developments

Continue the story:"""

        print(f"Generating Scene {scene_num}: '{title}' (target: {target_words} words)...")

        max_tokens = int(target_words * 1.15)
        temperature = 0.55

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.85,
                repeat_penalty=1.15,
                frequency_penalty=0.1,
                presence_penalty=0.05,
                stop=[
                    "###", "---", "The End", "Chapter", "CHAPTER", 
                    "\n\n\n", self.current_story_data['title'], "SCENE", "Scene",
                    "STORY:", "INSTRUCTIONS:", "MANDATORY ELEMENTS:",
                    "CRITICAL INSTRUCTIONS:", "Continue the story:"
                ],
                echo=False
            )

            scene_text = self._clean_and_format_text(output['choices'][0]['text'])
            word_count = len(scene_text.split())

            scene_summary = self._create_scene_summary(scene_text, scene_data)
            self.scene_summaries.append(scene_summary)

            if self._validate_scene_quality(scene_text, scene_data):
                print(f"Scene {scene_num} completed: {word_count} words (Health: {word_count/target_words:.1%})")
                return scene_text
            else:
                print(f"✗ Scene {scene_num} failed quality check (got {word_count} words). Returning partial content.")
                return scene_text

        except Exception as e:
            print(f"✗ Error during scene generation: {e}")
            return f"[Error: Could not generate Scene {scene_num}: {title}]"

    def _validate_scene_quality(self, scene_text: str, scene_data: Dict[str, str]) -> bool:
        """Validate scene quality beyond just word count."""
        if not scene_text or scene_text.startswith("[Error"):
            return False
        word_count = len(scene_text.split())
        target_words = scene_data["target_words"]
        if word_count / target_words < self.health_threshold:
            return False
        sentences = [s.strip() for s in scene_text.split('.') if s.strip()]
        if len(sentences) < 3:
            return False
        words = scene_text.lower().split()
        if len(words) > 0 and len(set(words)) / len(words) < 0.6:
            print(f"  Warning: Scene may be repetitive (uniqueness: {len(set(words))/len(words):.1%})")
        return True

    def _clean_and_format_text(self, text: str) -> str:
        """Enhanced text cleaning and formatting with better repetition removal."""
        if not text:
            return ""
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\*+|---+', '', text)
        unwanted_patterns = [
            r'^(SCENE \d+:.*?|INSTRUCTIONS:.*?|KEY ELEMENTS.*?|MANDATORY.*?|CRITICAL.*?|WRITING.*?|STORY:.*?|CURRENT SCENE:.*?)$',
            r'^(Continue the story:?|Begin the story:?|\.{3,}|_{3,}|-{3,})$',
            r'^[.*?]$',
            r'^(PREVIOUS PLOT DEVELOPMENTS:.*?)$'
        ]
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not any(skip_word in line.lower() for skip_word in 
                              ['scene', 'instruction', 'mandatory', 'critical', 'story:', 'continue the', 'begin the']):
                lines.append(line)
        formatted_lines = []
        prev_line = ""
        for line in lines:
            if line != prev_line:
                formatted_lines.append(line)
                prev_line = line
        formatted_text = '\n\n'.join(formatted_lines)
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text).strip()
        return formatted_text

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story with improved context management."""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")

        self.current_story_data = story_data
        self.scene_summaries = []

        story_structure = self.create_story_structure(story_data)
        full_story_text = ""
        story_scenes, total_words, failed_scenes = [], 0, 0
        outline_for_prompt = story_data.get('outline', '')

        for scene_data in story_structure:
            scene_text = self.generate_scene(scene_data, outline_for_prompt, full_story_text)
            if scene_text and not scene_text.startswith("[Error"):
                story_scenes.append({'title': scene_data['title'], 'text': scene_text})
                full_story_text += "\n\n" + scene_text
                scene_words = len(scene_text.split())
                total_words += scene_words
                print(f"  Progress: {total_words}/{self.total_target_words} words ({total_words/self.total_target_words*100:.1f}%)")
                if scene_data['scene'] % 2 == 0:
                    gc.collect()
            else:
                failed_scenes += 1
                print(f"✗ Scene {scene_data['scene']} failed to generate properly.")

        complete_story_text = self._final_story_cleanup("\n\n---\n\n".join([scene['text'] for scene in story_scenes]))
        story_health = total_words / self.total_target_words if self.total_target_words > 0 else 0

        print("\n=== Story Generation Complete ===")
        print(f"Final word count: {len(complete_story_text.split())} words ({story_health:.1%} of target)")

        return {
            'title': story_data['title'], 
            'text': complete_story_text, 
            'word_count': len(complete_story_text.split()),
            'target_words': self.total_target_words, 
            'health_ratio': story_health,
            'scenes_completed': len(story_scenes), 
            'scenes_failed': failed_scenes
        }

    def _final_story_cleanup(self, story_text: str) -> str:
        """Final cleanup of the complete story."""
        story_text = re.sub(r'---+', '---', story_text)
        story_text = re.sub(r'[.*?]', '', story_text)
        story_text = re.sub(r'\n{4,}', '\n\n\n', story_text)
        return story_text.strip()

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
    """Generate audio narration with improved error handling"""
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
        chunks = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
        if not chunks:
            print("No suitable text to narrate.")
            return None
        print(f"Narrating {len(chunks)} paragraphs...")
        audio_segments = []
        for i, chunk in enumerate(chunks):
            try:
                if len(chunk) > 500:
                    chunk = chunk[:497] + "..."
                results = list(pipeline(chunk, voice='bm_fable'))
                for _, _, audio in results:
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        if audio_segments:
            try:
                combined_audio = np.concatenate(audio_segments, axis=0)
                sf.write(output_file, combined_audio, 24000)
                duration = len(combined_audio) / 24000
                print(f"Audio saved: '{output_file}' (Duration: {duration/60:.2f} minutes)")
                return output_file
            except Exception as e:
                print(f"Error combining audio segments: {e}")
                return None
        else:
            print("No audio segments were successfully generated.")
            return None
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return None

def send_to_telegram(bot_token: str, chat_id: str, text_file: str = None, audio_file: str = None):
    """Send files to Telegram with improved error handling"""
    if not bot_token or not chat_id:
        print("Telegram credentials not provided. Skipping upload.")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    for file_type, file_path in [('story', text_file), ('audio', audio_file)]:
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            print(f"Uploading {file_type} to Telegram...")
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                print(f"✗ {file_type} file too large ({file_size/1024/1024:.1f}MB)")
                continue
            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {'chat_id': chat_id}
                response = requests.post(url, data=data, files=files, timeout=300)
                if response.status_code == 200:
                    print(f"✓ Successfully sent {file_type}.")
                else:
                    print(f"✗ Failed to send {file_type}: {response.text}")
        except Exception as e:
            print(f"✗ Error sending {file_type}: {e}")

def main(story_outline_str: str):
    """Main execution function with improved error handling"""
    print("=== OPTIMIZED HORROR STORY GENERATOR ===")
    try:
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
                print(f"Story: {story['title']}")
                print(f"Final word count: {story['word_count']}")
                print(f"Scenes completed: {story['scenes_completed']}")
        else:# The Soshanguve Stalker  

## Story Outline  
Pretoria, present day. The community of Soshanguve is paralyzed with fear after a string of horrifying attacks on young women. Over the course of several months, multiple victims are abducted, assaulted, and left for dead in isolated areas. Each attack follows the same disturbing pattern: the victims are found with their hands bound and a crude, hand-carved message scrawled into the ground nearby—**"Forgive me."**  

Detective Lerato Phiri, a rising star in the police force, is assigned to the case after public outcry about the lack of progress. The media labels the perpetrator "The Soshanguve Stalker," stoking panic in an already-frightened community. Despite her determination, Lerato struggles to make headway as the perpetrator leaves no forensic evidence behind.  

As Lerato investigates, she uncovers a deeply disturbing connection between the victims: they were all involved in exposing a powerful local pastor, a man who was accused of sexual abuse and financial corruption but never faced justice due to community support and bribed law enforcement. The women had banded together as part of a survivors' group but had been forced underground after receiving threats.  

The investigation takes a shocking turn when Lerato discovers that the stalker is not connected to the pastor but is, in fact, one of his most ardent followers—a man who sees himself as an "avenging angel" sent to silence the women for tarnishing the pastor's name. Lerato finds herself racing against time to stop the stalker and expose the pastor’s crimes, all while navigating the dangerous political and social tensions in the community.  

The horror lies not just in the stalker’s sadistic methods but in the systemic injustice that allowed the original crimes to go unpunished, leaving the survivors vulnerable to further violence.  

## Key Characters  
- **Detective Lerato Phiri:** Early 30s, fiercely independent and driven by a strong sense of justice. She is determined to prove herself in a male-dominated field while struggling with the emotional toll of investigating such brutal crimes.  
- **The Soshanguve Stalker:** A fanatical supporter of the disgraced pastor, whose twisted sense of morality drives them to commit the heinous attacks.  
- **Nomvula Mokoena:** A survivor of the pastor’s abuse and an outspoken activist. She becomes Lerato’s key ally, but her own safety is at constant risk as the stalker escalates their attacks.  

## Setting  
- **Soshanguve, Pretoria:** The story is set in the densely populated township of Soshanguve, where poverty and crime create a tense and volatile backdrop. The stark contrast between the community’s faith-driven culture and the dark secrets lurking beneath adds to the story’s unsettling atmosphere.  
- **Time Period:** Present day, with the central crimes reflecting the real-life struggles of survivors seeking justice in a system that often fails them.  
            print("\nERROR: Failed to generate a story with any content.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # NEW Jack Ketchum-style Story Prompt for a contemporary Pretoria crime
    jack_ketchum_story_outline = """
# The Soshanguve Stalker  
## Story Outline  
Pretoria, present day. The community of Soshanguve is paralyzed with fear after a string of horrifying attacks on young women. Over the course of several months, multiple victims are abducted, assaulted, and left for dead in isolated areas. Each attack follows the same disturbing pattern: the victims are found with their hands bound and a crude, hand-carved message scrawled into the ground nearby—**"Forgive me."**  

Detective Lerato Phiri, a rising star in the police force, is assigned to the case after public outcry about the lack of progress. The media labels the perpetrator "The Soshanguve Stalker," stoking panic in an already-frightened community. Despite her determination, Lerato struggles to make headway as the perpetrator leaves no forensic evidence behind.  

As Lerato investigates, she uncovers a deeply disturbing connection between the victims: they were all involved in exposing a powerful local pastor, a man who was accused of sexual abuse and financial corruption but never faced justice due to community support and bribed law enforcement. The women had banded together as part of a survivors' group but had been forced underground after receiving threats.  

The investigation takes a shocking turn when Lerato discovers that the stalker is not connected to the pastor but is, in fact, one of his most ardent followers—a man who sees himself as an "avenging angel" sent to silence the women for tarnishing the pastor's name. Lerato finds herself racing against time to stop the stalker and expose the pastor’s crimes, all while navigating the dangerous political and social tensions in the community.  

The horror lies not just in the stalker’s sadistic methods but in the systemic injustice that allowed the original crimes to go unpunished, leaving the survivors vulnerable to further violence.  

## Key Characters  
- **Detective Lerato Phiri:** Early 30s, fiercely independent and driven by a strong sense of justice. She is determined to prove herself in a male-dominated field while struggling with the emotional toll of investigating such brutal crimes.  
- **The Soshanguve Stalker:** A fanatical supporter of the disgraced pastor, whose twisted sense of morality drives them to commit the heinous attacks.  
- **Nomvula Mokoena:** A survivor of the pastor’s abuse and an outspoken activist. She becomes Lerato’s key ally, but her own safety is at constant risk as the stalker escalates their attacks.  

## Setting  
- **Soshanguve, Pretoria:** The story is set in the densely populated township of Soshanguve, where poverty and crime create a tense and volatile backdrop. The stark contrast between the community’s faith-driven culture and the dark secrets lurking beneath adds to the story’s unsettling atmosphere.  
- **Time Period:** Present day, with the central crimes reflecting the real-life struggles of survivors seeking justice in a system that often fails them.  
""".replace("-", "") 

    main(jack_ketchum_story_outline)