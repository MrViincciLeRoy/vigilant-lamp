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
        self.scene_summaries = []  # NEW: Track scene summaries for better context

        # MODIFICATION: Drastically reduced total word count to make it feasible on a GitHub runner.
        self.total_target_words = 2800
        self.health_threshold = 0.75  # 75% minimum word generation health is more forgiving

    def create_story_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create a detailed story structure with scenes and plot points based on the story outline.
        """
        return [
            {
                "scene": 1, "title": "The First Stain",
                "description": "Introduce Detective Naledi as she arrives at a horrifying crime scene in a seemingly peaceful Pretoria suburb.",
                "target_words": 400, 
                "key_elements": ["Detective Naledi introduction", "Brutal crime scene details (implied, not explicit gore)", "Initial shock and questions", "Setting the grim tone in Pretoria"],
                "focus": "character_introduction"
            },
            {
                "scene": 2, "title": "Whispers from the Past",
                "description": "Naledi uncovers a cold case from years ago with unsettling similarities, suggesting a serial pattern.",
                "target_words": 400, 
                "key_elements": ["Discovery of old files/connection", "Similar modus operandi", "Initial dismissal by colleagues", "Naledi's growing unease"],
                "focus": "investigation_begins"
            },
            {
                "scene": 3, "title": "A Trail of Breadcrumbs",
                "description": "The killer leaves cryptic clues, drawing Naledi deeper into their twisted game.",
                "target_words": 400, 
                "key_elements": ["Cryptic messages or symbols at crime scenes", "Killer taunting the police/Naledi directly", "Naledi follows a lead that takes her to a dark place", "Rising tension"],
                "focus": "clues_and_escalation"
            },
            {
                "scene": 4, "title": "The Shadow Falls",
                "description": "The killer escalates, targeting someone close to Naledi or leaving a particularly disturbing message.",
                "target_words": 400, 
                "key_elements": ["Personal threat or near-miss for Naledi", "Heightened sense of danger", "A chilling realization about the killer's motive", "Isolation of Naledi"],
                "focus": "personal_stakes"
            },
            {
                "scene": 5, "title": "Beneath the Surface",
                "description": "Naledi discovers a horrific secret about Pretoria's dark underbelly connected to the killer's motives.",
                "target_words": 400, 
                "key_elements": ["Uncovering a hidden network or dark history", "Motive is more twisted than initially thought", "Moral compromises or shocking revelations", "Naledi's determination amidst despair"],
                "focus": "revelation"
            },
            {
                "scene": 6, "title": "The Unveiling",
                "description": "The climactic confrontation between Naledi and the killer, revealing their identity and the full horror.",
                "target_words": 400, 
                "key_elements": ["Climactic showdown at a desolate location", "Killer's identity exposed", "Twisted philosophy or justification", "Brutal physical and psychological struggle"],
                "focus": "climax"
            },
            {
                "scene": 7, "title": "Lingering Scars",
                "description": "The aftermath, exploring the psychological toll on Naledi and the lasting impact on Pretoria.",
                "target_words": 400, 
                "key_elements": ["Resolution of the main conflict", "Naledi's trauma and changed outlook", "Pretoria grappling with the unearthed horrors", "A somber, unsettling peace"],
                "focus": "resolution"
            }
        ]

    def _create_scene_summary(self, scene_text: str, scene_data: Dict[str, str]) -> str:
        """Create a concise summary of a scene for context in future scenes."""
        # Extract key plot points and character actions
        sentences = [s.strip() for s in scene_text.split('.') if s.strip()]
        
        # Focus on the last few sentences which usually contain the most important developments
        key_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
        
        summary = f"Scene {scene_data['scene']} ({scene_data['title']}): " + '. '.join(key_sentences)
        
        # Limit summary length to prevent context bloat
        if len(summary) > 200:
            summary = summary[:197] + "..."
            
        return summary

    def _build_focused_context(self, scene_num: int) -> str:
        """Build focused context from previous scene summaries."""
        if not self.scene_summaries:
            return ""
        
        # For early scenes, use more context. For later scenes, focus on recent developments
        if scene_num <= 3:
            relevant_summaries = self.scene_summaries
        else:
            # Use last 2 scenes for context to prevent overwhelming the model
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

        # NEW: Build focused context from scene summaries
        focused_context = self._build_focused_context(scene_num)

        # NEW: Create progression instructions based on scene focus
        progression_instructions = {
            "character_introduction": "Focus on establishing character depth and the immediate situation. Avoid backstory dumps.",
            "investigation_begins": "Advance the investigation with new discoveries. Show detective work in action.",
            "clues_and_escalation": "Introduce new evidence and raise the stakes. Build tension through discoveries.",
            "personal_stakes": "Make the threat personal and immediate. Show character vulnerability.",
            "revelation": "Reveal crucial information that changes everything. Focus on the impact of discovery.",
            "climax": "Bring all elements together in confrontation. Focus on action and revelation.",
            "resolution": "Show consequences and aftermath. Avoid tying up loose ends too neatly."
        }

        # NEW: Improved prompting with stronger continuation guidance
        if scene_num == 1:
            prompt = f"""You are writing a brutal, realistic horror story in the style of Jack Ketchum. Write the opening scene with visceral impact and psychological depth.

STORY: {story_data['title']}
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
            prompt = f"""Continue the horror story "{story_data['title']}" with the next scene. Ensure narrative progression and avoid repetition.

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
        
        # NEW: More conservative generation parameters to reduce repetition
        max_tokens = int(target_words * 1.15)  # Tighter control
        temperature = 0.55  # Lower temperature for more focused output
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.85,  # More focused sampling
                repeat_penalty=1.15,  # Higher penalty for repetition
                frequency_penalty=0.1,  # NEW: Penalize frequent words
                presence_penalty=0.05,  # NEW: Encourage new concepts
                stop=[
                    "###", "---", "The End", "Chapter", "CHAPTER", 
                    "\n\n\n", story_data['title'], "SCENE", "Scene",
                    "STORY:", "INSTRUCTIONS:", "MANDATORY ELEMENTS:",
                    "CRITICAL INSTRUCTIONS:", "Continue the story:"
                ],
                echo=False
            )
            
            scene_text = self._clean_and_format_text(output['choices'][0]['text'])
            word_count = len(scene_text.split())

            # NEW: Create and store scene summary for future context
            scene_summary = self._create_scene_summary(scene_text, scene_data)
            self.scene_summaries.append(scene_summary)

            # NEW: Validate scene quality beyond just word count
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
        
        # Check word count health
        if word_count / target_words < self.health_threshold:
            return False
            
        # NEW: Check for repetitive patterns
        sentences = [s.strip() for s in scene_text.split('.') if s.strip()]
        if len(sentences) < 3:  # Too short to be meaningful
            return False
            
        # Check for excessive repetition of words (simple heuristic)
        words = scene_text.lower().split()
        if len(set(words)) / len(words) < 0.6:  # Less than 60% unique words suggests repetition
            print(f"  Warning: Scene may be repetitive (uniqueness: {len(set(words))/len(words):.1%})")
            
        return True

    def _clean_and_format_text(self, text: str) -> str:
        """Enhanced text cleaning and formatting with better repetition removal."""
        if not text:
            return ""
            
        # Remove code blocks and formatting
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\*+|---+', '', text)
        
        # NEW: More aggressive cleaning of unwanted meta-text
        unwanted_patterns = [
            r'^(SCENE \d+:.*?|INSTRUCTIONS:.*?|KEY ELEMENTS.*?|MANDATORY.*?|CRITICAL.*?|WRITING.*?|STORY:.*?|CURRENT SCENE:.*?)$',
            r'^(Continue the story:?|Begin the story:?|\.{3,}|_{3,}|-{3,})$',
            r'^\[.*?\]$',  # Remove anything in brackets
            r'^(PREVIOUS PLOT DEVELOPMENTS:.*?)$'
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up lines
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not any(skip_word in line.lower() for skip_word in 
                              ['scene', 'instruction', 'mandatory', 'critical', 'story:', 'continue the', 'begin the']):
                lines.append(line)
        
        # NEW: Remove duplicate consecutive sentences
        formatted_lines = []
        prev_line = ""
        for line in lines:
            if line != prev_line:  # Simple duplicate removal
                formatted_lines.append(line)
                prev_line = line
        
        formatted_text = '\n\n'.join(formatted_lines)
        
        # Clean up excessive whitespace
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text).strip()
        
        return formatted_text

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story with improved context management."""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")

        # Reset context for new story
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
                
                # NEW: More aggressive memory management
                if scene_data['scene'] % 2 == 0:  # Every other scene
                    gc.collect()
            else:
                failed_scenes += 1
                print(f"✗ Scene {scene_data['scene']} failed to generate properly.")

        # NEW: Final story cleanup
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
        # Remove excessive scene separators
        story_text = re.sub(r'---+', '---', story_text)
        
        # Remove any remaining meta-text that might have slipped through
        story_text = re.sub(r'\[.*?\]', '', story_text)
        
        # Clean up spacing
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

        # Process text in manageable chunks
        chunks = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
        if not chunks:
            print("No suitable text to narrate.")
            return None

        print(f"Narrating {len(chunks)} paragraphs...")
        
        audio_segments = []
        for i, chunk in enumerate(chunks):
            try:
                # Limit chunk size to prevent memory issues
                if len(chunk) > 500:
                    chunk = chunk[:497] + "..."
                    
                results = list(pipeline(chunk, voice='bm_fable'))
                for _, _, audio in results:
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                        
                # Progress indicator
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
            
            # Check file size (Telegram has limits)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
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
        else:
            print("\nERROR: Failed to generate a story with any content.")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Jack Ketchum-style Story Prompt for Pretoria Crimes
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

    main(jack_ketchum_story_outline)