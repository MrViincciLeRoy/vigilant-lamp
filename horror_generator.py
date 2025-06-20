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
import argparse
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
        # Extract key information from the story outline
        outline = story_data.get('outline', '')
        title = story_data.get('title', 'Untitled Horror Story')
        
        # Parse the outline to determine story structure
        scenes = self._analyze_outline_for_scenes(outline, title)
        
        # If we can't parse specific scenes, use a generic 7-act structure
        if len(scenes) < 5:
            scenes = self._create_generic_structure(story_data)
        
        return scenes
    
    def _analyze_outline_for_scenes(self, outline: str, title: str) -> List[Dict[str, str]]:
        """Analyze the outline to determine natural scene breaks"""
        scenes = []
        
        # Look for common story structure indicators
        outline_lower = outline.lower()
        
        # Try to identify key story beats from the outline
        if any(word in outline_lower for word in ['murder', 'kill', 'death', 'ritual']):
            # Crime/horror story structure
            scenes = [
                {
                    "scene": 1,
                    "title": "The Setup",
                    "description": "Introduce main characters and establish the normal world before darkness intrudes",
                    "target_words": 1400,
                    "key_elements": ["Character introduction", "Setting establishment", "Initial relationships"]
                },
                {
                    "scene": 2,
                    "title": "The Catalyst",
                    "description": "The event or meeting that sets the horror in motion",
                    "target_words": 1400,
                    "key_elements": ["Inciting incident", "First hint of danger", "Character motivations revealed"]
                },
                {
                    "scene": 3,
                    "title": "Rising Tension",
                    "description": "Escalating conflict and growing unease among characters",
                    "target_words": 1400,
                    "key_elements": ["Character development", "Relationship strain", "Building suspense"]
                },
                {
                    "scene": 4,
                    "title": "The Point of No Return",
                    "description": "Characters make crucial decisions that seal their fate",
                    "target_words": 1400,
                    "key_elements": ["Major plot development", "Character choices", "Moral conflict"]
                },
                {
                    "scene": 5,
                    "title": "The Descent",
                    "description": "Characters move toward their doom, tension reaches breaking point",
                    "target_words": 1400,
                    "key_elements": ["Final preparations", "Last chances", "Dramatic irony"]
                },
                {
                    "scene": 6,
                    "title": "The Horror Unleashed",
                    "description": "The climactic horror scene where the worst happens",
                    "target_words": 1400,
                    "key_elements": ["Climax", "Horror peak", "Character fates sealed"]
                },
                {
                    "scene": 7,
                    "title": "Aftermath and Resolution",
                    "description": "The consequences unfold and justice or closure is achieved",
                    "target_words": 1600,
                    "key_elements": ["Resolution", "Consequences", "Final justice"]
                }
            ]
        
        return scenes
    
    def _create_generic_structure(self, story_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Create a generic horror story structure when outline analysis fails"""
        return [
            {
                "scene": 1,
                "title": "The Beginning",
                "description": "Establish characters, setting, and the world before horror intrudes",
                "target_words": 1400,
                "key_elements": ["Character introduction", "Setting", "Normal world"]
            },
            {
                "scene": 2,
                "title": "First Shadows",
                "description": "Something dark enters the story, first hints of the horror to come",
                "target_words": 1400,
                "key_elements": ["Inciting incident", "First supernatural/horror element", "Character reactions"]
            },
            {
                "scene": 3,
                "title": "Growing Darkness",
                "description": "The horror elements strengthen, characters begin to change",
                "target_words": 1400,
                "key_elements": ["Horror escalation", "Character development", "Building tension"]
            },
            {
                "scene": 4,
                "title": "The Turning Point",
                "description": "A crucial moment that changes everything for the characters",
                "target_words": 1400,
                "key_elements": ["Major plot twist", "Character revelations", "Point of no return"]
            },
            {
                "scene": 5,
                "title": "Into the Abyss",
                "description": "Characters descend deeper into horror, hope begins to fade",
                "target_words": 1400,
                "key_elements": ["Desperation", "Horror intensifies", "Character struggles"]
            },
            {
                "scene": 6,
                "title": "The Heart of Darkness",
                "description": "The climactic horror scene, the darkest moment of the story",
                "target_words": 1400,
                "key_elements": ["Climax", "Ultimate horror", "Character fates"]
            },
            {
                "scene": 7,
                "title": "What Remains",
                "description": "The aftermath, consequences, and any resolution or closure",
                "target_words": 1600,
                "key_elements": ["Resolution", "Aftermath", "Final thoughts"]
            }
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

        # Build comprehensive context from previous scenes
        context_length = min(6000, len(self.story_context))  # Use substantial context
        recent_context = self.story_context[-context_length:] if self.story_context else ""
        
        # Create scene-specific instructions
        if scene_num == 1:
            prompt = f"""Write the opening scene of a psychological horror story.

STORY CONTEXT:
{story_context}

SCENE: {title}
DESCRIPTION: {description}
TARGET LENGTH: {target_words} words
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}

WRITING REQUIREMENTS:
- Write in third person with rich character perspectives
- Use rich sensory details and atmospheric descriptions
- Build psychological tension through character interactions
- Show character motivations through actions and dialogue
- Maintain a dark, foreboding tone throughout
- Create vivid, cinematic scenes
- Write exactly {target_words} words
- Establish the setting and characters based on the story context provided

Begin the story:"""

        else:
            # Continuation scenes with full context
            prompt = f"""Continue this psychological horror story.

STORY CONTEXT (for reference):
{story_context}

PREVIOUS STORY CONTENT:
{recent_context}

CURRENT SCENE: {title}
SCENE DESCRIPTION: {description}
TARGET LENGTH: {target_words} words
KEY ELEMENTS TO INCLUDE: {', '.join(key_elements)}

WRITING REQUIREMENTS:
- Maintain perfect narrative continuity with previous scenes
- Keep character voices and personalities consistent
- Escalate psychological tension appropriately for scene {scene_num}
- Use rich sensory details and atmospheric descriptions
- Show rather than tell - use dialogue and action
- Build toward the climactic scenes appropriately
- Write exactly {target_words} words
- End with a natural transition point for the next scene

Continue the story seamlessly:"""

        print(f"Generating Scene {scene_num}: '{title}' (target: {target_words} words)...")
        
        # Calculate tokens with buffer for full context usage
        max_tokens = int(target_words * 1.8)  # Generous token allowance
        
        # Multiple generation attempts for quality
        best_attempt = ""
        best_word_count = 0
        
        for attempt in range(3):  # Try up to 3 times
            try:
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature + (attempt * 0.05),  # Slight temperature variation
                    top_p=top_p,
                    repeat_penalty=1.05 + (attempt * 0.02),
                    stop=["###", "---", "THE END", "Chapter", "CHAPTER", "\n\n\n\n"],
                    echo=False
                )
                
                scene_text = output['choices'][0]['text'].strip()
                scene_text = self._clean_and_format_text(scene_text)
                word_count = len(scene_text.split())
                
                # Check if this attempt is better (closer to target)
                target_proximity = abs(word_count - target_words) / target_words
                best_proximity = abs(best_word_count - target_words) / target_words if best_word_count > 0 else 1.0
                
                if target_proximity < best_proximity and word_count >= (target_words * 0.7):
                    best_attempt = scene_text
                    best_word_count = word_count
                
                # If we hit the health threshold, use this attempt
                health_ratio = word_count / target_words
                if health_ratio >= self.health_threshold:
                    best_attempt = scene_text
                    best_word_count = word_count
                    break
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_attempt:
            health_ratio = best_word_count / target_words
            print(f"Scene {scene_num} completed: {best_word_count} words (Health: {health_ratio:.1%})")
            return best_attempt
        else:
            print(f"Failed to generate Scene {scene_num}")
            return f"[Error: Could not generate Scene {scene_num}: {title}]"

    def _clean_and_format_text(self, text: str) -> str:
        """Enhanced text cleaning and formatting"""
        if not text:
            return ""
            
        # Remove unwanted formatting
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\*+', '', text)
        text = re.sub(r'---+', '', text)
        
        # Fix paragraph spacing
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('SCENE'):
                cleaned_lines.append(line)
        
        # Join with proper paragraph breaks
        formatted_text = '\n\n'.join(cleaned_lines)
        
        # Remove any remaining artifacts
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        
        return formatted_text.strip()

    def generate_complete_story(self, story_data: Dict[str, str]) -> Dict[str, str]:
        """Generate a complete cohesive horror story with improved structure"""
        if not self.model_loaded:
            raise ValueError("Model not loaded - cannot generate story")
        
        print(f"=== Generating Complete Horror Story: '{story_data['title']}' ===")
        print(f"Target: {self.total_target_words} words | Health Threshold: {self.health_threshold:.0%}")
        
        # Create detailed story structure
        story_structure = self.create_story_structure(story_data)
        
        story_scenes = []
        total_words = 0
        failed_scenes = 0
        
        for scene_data in story_structure:
            scene_text = self.generate_scene(scene_data, story_structure, story_data.get('outline', ''))
            
            if scene_text and not scene_text.startswith("[Error"):
                story_scenes.append({
                    'title': scene_data['title'],
                    'text': scene_text
                })
                
                # Update context with new scene
                self.story_context += "\n\n" + scene_text
                
                scene_words = len(scene_text.split())
                total_words += scene_words
                
                health_ratio = scene_words / scene_data['target_words']
                print(f"✓ Scene {scene_data['scene']} added: {scene_words} words")
                print(f"  Progress: {total_words}/{self.total_target_words} words ({total_words/self.total_target_words*100:.1f}%)")
                print(f"  Scene Health: {health_ratio:.1%}")
                
                # Memory management
                gc.collect()
                
            else:
                failed_scenes += 1
                print(f"✗ Scene {scene_data['scene']} failed")
        
        # Join all scenes into complete story
        complete_story = ""
        for i, scene in enumerate(story_scenes):
            if i > 0:
                complete_story += "\n\n---\n\n"
            complete_story += scene['text']
        
        # Calculate final statistics
        story_health = total_words / self.total_target_words
        
        print(f"\n=== Story Generation Complete ===")
        print(f"Scenes generated: {len(story_scenes)}/{len(story_structure)}")
        print(f"Failed scenes: {failed_scenes}")
        print(f"Final word count: {total_words} words")
        print(f"Target achievement: {story_health:.1%}")
        print(f"Health status: {'✓ HEALTHY' if story_health >= self.health_threshold else '⚠ BELOW THRESHOLD'}")
        
        return {
            'title': story_data['title'],
            'text': complete_story,
            'word_count': total_words,
            'target_words': self.total_target_words,
            'health_ratio': story_health,
            'scenes_completed': len(story_scenes),
            'scenes_failed': failed_scenes
        }

def read_markdown_file(file_path: str) -> str:
    """Reads markdown file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return ""

def extract_story_data(markdown_text: str) -> Dict[str, str]:
    """Extract story data from markdown - now more flexible"""
    lines = markdown_text.split('\n')
    story_title = ""
    
    # Look for title in various formats
    for line in lines:
        line = line.strip()
        if line.startswith('# ') and not story_title:
            story_title = line[2:].strip()
            break
        elif line.startswith('## ') and 'title' in line.lower() and not story_title:
            # Look for title in next line
            idx = lines.index(line)
            if idx + 1 < len(lines):
                story_title = lines[idx + 1].strip()
                break
    
    return {
        'title': story_title or 'Untitled Horror Story',
        'outline': markdown_text.strip()
    }

def save_story(story: Dict[str, str], output_dir: str = 'outputs') -> Optional[str]:
    """Save complete story with enhanced metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from title
    safe_title = re.sub(r'[^\w\s-]', '', story['title']).strip()
    safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()
    filename = os.path.join(output_dir, f"{safe_title}_complete.md")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {story['title']}\n\n")
            f.write(story['text'])
            f.write(f"\n\n---\n\n")
            f.write("## Story Statistics\n\n")
            f.write(f"- **Word Count:** {story['word_count']:,} words\n")
            f.write(f"- **Target:** {story['target_words']:,} words\n")
            f.write(f"- **Achievement:** {story['health_ratio']:.1%}\n")
            f.write(f"- **Scenes Completed:** {story['scenes_completed']}\n")
            f.write(f"- **Scenes Failed:** {story['scenes_failed']}\n")
            f.write(f"- **Health Status:** {'✓ HEALTHY' if story['health_ratio'] >= 0.8 else '⚠ BELOW THRESHOLD'}\n")
        
        print(f"Story saved: {filename}")
        print(f"Final statistics: {story['word_count']:,} words ({story['health_ratio']:.1%} of target)")
        return filename
    except Exception as e:
        print(f"Error saving story: {e}")
        return None

def generate_audio(text: str, output_dir: str = 'outputs', story_title: str = 'story') -> Optional[str]:
    """Generate audio narration with better error handling"""
    try:
        print("Generating audio narration...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from title
        safe_title = re.sub(r'[^\w\s-]', '', story_title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()
        output_file = os.path.join(output_dir, f"{safe_title}_narration.wav")
        
        # Check if kokoro is available
        try:
            pipeline = kokoro.KPipeline(lang_code='a')
        except Exception as e:
            print(f"Kokoro not available: {e}")
            return None
        
        audio_segments = []
        
        # Split text into manageable chunks (by paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 500:  # Split long paragraphs
                sentences = paragraph.split('. ')
                for j in range(0, len(sentences), 3):  # 3 sentences per chunk
                    chunk = '. '.join(sentences[j:j+3])
                    if chunk:
                        try:
                            generator = pipeline(chunk, voice='bm_fable')
                            for _, _, audio in generator:
                                if audio is not None and len(audio) > 0:
                                    audio_segments.append(audio)
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
            else:
                try:
                    generator = pipeline(paragraph, voice='bm_fable')
                    for _, _, audio in generator:
                        if audio is not None and len(audio) > 0:
                            audio_segments.append(audio)
                except Exception as e:
                    print(f"Error processing paragraph: {e}")
            
            if i % 10 == 0:  # Progress update
                print(f"Audio progress: {i}/{total_paragraphs} paragraphs processed")

        if audio_segments:
            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            print(f"Audio saved: '{output_file}' (Duration: {duration/60:.2f} minutes)")
            return output_file
        else:
            print("No audio segments generated")
            return None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def send_to_telegram(bot_token: str, chat_id: str, text_file: str = None, audio_file: str = None):
    """Send files to Telegram with better error handling"""
    if not bot_token or not chat_id:
        print("Telegram credentials not provided (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    
    files_to_send = []
    if text_file and os.path.exists(text_file):
        files_to_send.append(('story', text_file))
    if audio_file and os.path.exists(audio_file):
        files_to_send.append(('audio', audio_file))
    
    for file_type, file_path in files_to_send:
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"Uploading {file_type} ({file_size:.1f} MB) to Telegram...")
            
            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {'chat_id': chat_id}
                response = requests.post(url, data=data, files=files, timeout=300)
                
                if response.status_code == 200:
                    print(f"✓ Successfully sent {file_type} to Telegram")
                else:
                    print(f"✗ Failed to send {file_type}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ Error sending {file_type}: {e}")

def create_example_outline():
    """Create an example story outline file"""
    example_content = """# The Cursed Inheritance

## Story Outline
When Sarah inherits her grandmother's Victorian mansion, she discovers it comes with more than just dusty furniture and old photographs. Hidden in the walls are journals detailing a century of dark family secrets, ritualistic practices, and a curse that has claimed the life of every woman in her bloodline on their 30th birthday.

As Sarah's 30th birthday approaches, strange occurrences begin to plague the house. Shadows move independently, whispers echo through empty rooms, and she begins experiencing vivid nightmares of her ancestors' deaths. She learns that her grandmother tried to break the curse but failed, becoming its latest victim.

The story follows Sarah's desperate race against time as she unravels the mystery of the family curse, confronts the malevolent spirit that enforces it, and must decide whether to sacrifice herself to save future generations or find another way to break the cycle of death that has haunted her family for over a century.

## Key Characters
- Sarah Mitchell: 29-year-old protagonist, inherits the cursed mansion
- Eleanor Whitmore: Sarah's deceased grandmother, previous victim of the curse
- The Shadow Entity: Malevolent spirit that enforces the family curse
- Marcus Chen: Local historian who helps Sarah research her family history
- Dr. Rebecca Torres: Paranormal investigator Sarah contacts for help

## Setting
- The Whitmore Victorian Mansion: A sprawling, isolated house built in 1890
- Millbrook, a small New England town with a dark history
- Time period: Present day, with flashbacks to previous generations
"""
    
    with open('example_story_outline.md', 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    print("Example story outline created: example_story_outline.md")
    return 'example_story_outline.md'

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Generic Horror Story Generator')
    parser.add_argument('--input', '-i', type=str, help='Input markdown file with story outline')
    parser.add_argument('--output', '-o', type=str, default='outputs', help='Output directory')
    parser.add_argument('--create-example', action='store_true', help='Create an example story outline')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio generation')
    parser.add_argument('--no-telegram', action='store_true', help='Skip Telegram upload')
    
    args = parser.parse_args()
    
    print("=== GENERIC HORROR STORY GENERATOR ===")
    print("Target: 10,000 words | Health threshold: 80%")
    print("Using full model context (8192 tokens)")
    
    # Create example if requested
    if args.create_example:
        create_example_outline()
        return
    
    # Determine input file
    input_file = args.input
    if not input_file:
        # Look for common input file names
        possible_files = ['story_outline.md', 'outline.md', 'story.md', 'input.md']
        for filename in possible_files:
            if os.path.exists(filename):
                input_file = filename
                print(f"Found input file: {input_file}")
                break
        
        if not input_file:
            print("No input file specified. Use --input to specify a file, or --create-example to create a template.")
            return
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    generator = GenericHorrorGenerator()
    
    if not generator.model_loaded:
        print("ERROR: Model not loaded. Please check your setup.")
        return
    
    # Load story data
    markdown_text = read_markdown_file(input_file)
    story_data = extract_story_data(markdown_text)
    
    print(f"\nGenerating story: '{story_data['title']}'")
    print(f"Input file: {input_file}")
    print(f"Output directory: {args.output}")
    
    # Generate the complete story
    print("\n=== STORY GENERATION ===")
    story = generator.generate_complete_story(story_data)
    
    # Save the story
    story_file = save_story(story, args.output)
    
    if story_file:
        print_final_report(story)
        
        # Generate audio if story is healthy and not disabled
        audio_file = None
        if not args.no_audio and story['health_ratio'] >= 0.8:
            print("\n=== AUDIO GENERATION ===")
            audio_file = generate_audio(story['text'], args.output, story['title'])
        elif args.no_audio:
            print("\n=== SKIPPING AUDIO (disabled) ===")
        else:
            print("\n=== SKIPPING AUDIO (Story below health threshold) ===")
        
        # Send to Telegram if configured and not disabled
        if not args.no_telegram:
            print("\n=== TELEGRAM UPLOAD ===")
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            send_to_telegram(bot_token, chat_id, story_file, audio_file)
        else:
            print("\n=== SKIPPING TELEGRAM (disabled) ===")
        
        print("\n=== PROCESS COMPLETE ===")
        print(f"Story generated: {story_file}")
        if audio_file:
            print(f"Audio generated: {audio_file}")
    else:
        print("ERROR: Failed to save story")

def print_final_report(story: Dict[str, str]):
    """Print a detailed final report"""
    print(f"\n{'='*50}")
    print(f"FINAL STORY REPORT")
    print(f"{'='*50}")
    print(f"Title: {story['title']}")
    print(f"Word Count: {story['word_count']:,} words")
    print(f"Target: {story['target_words']:,} words")
    print(f"Achievement: {story['health_ratio']:.1%}")
    print(f"Health Status: {'✓ HEALTHY' if story['health_ratio'] >= 0.8 else '⚠ BELOW THRESHOLD'}")
    print(f"Scenes Completed: {story['scenes_completed']}")
    print(f"Scenes Failed: {story['scenes_failed']}")
    
    if story['health_ratio'] >= 0.8:
        print("✓ Story meets quality standards")
    else:
        print("⚠ Story below health threshold - consider regeneration")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()