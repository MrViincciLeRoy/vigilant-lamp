# -*- coding: utf-8 -*-
"""Optimized Horror Story Generator - Single Cohesive 10k Word Story"""
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import Dict, List
import gc
import kokoro
import soundfile as sf
import numpy as np
import requests

# Initialize the model
try:
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=4096,  # Increased context window
        verbose=False
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

class OptimizedHorrorGenerator:
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.story_context = ""
        self.character_names = []
        self.location_details = []

    def generate_story_segment(self, story_data: Dict[str, str], segment_num: int, 
                             target_words: int = 1500, temperature: float = 0.8, 
                             top_p: float = 0.95, repeat_penalty: float = 1.1) -> str:
        """Generate a single story segment that flows naturally from previous content"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        title = story_data['title']
        outline = story_data['outline']
        
        # Get the last 1000 characters for context
        recent_context = self.story_context[-1000:] if self.story_context else ""
        
        if segment_num == 1:
            # Opening segment
            prompt = f"""Write the opening of a psychological horror story titled "{title}".

Story outline: {outline}

Create a gripping opening that:
- Introduces the main characters naturally through action and dialogue
- Establishes the dark, unsettling atmosphere
- Sets up the central conflict or mystery
- Uses vivid, sensory descriptions
- Builds tension from the first paragraph
- Is approximately {target_words} words

Write in the style of modern psychological horror - visceral, emotionally intense, with rich character development. Focus on showing rather than telling, using dialogue and action to reveal character motivations.

Begin the story:"""

        else:
            # Continuation segments
            prompt = f"""Continue this psychological horror story titled "{title}". 

Previous context:
{recent_context}

Continue the story by:
- Maintaining narrative flow and character consistency
- Escalating the tension and horror elements
- Developing the plot based on the established setup
- Using vivid, sensory descriptions
- Keeping the pacing tight and engaging
- Writing approximately {target_words} words

Continue the story seamlessly:"""

        print(f"Generating segment {segment_num} (target: {target_words} words)...")
        
        # Calculate tokens (roughly 4 characters per token)
        max_tokens = int(target_words * 1.5)  # Give some buffer
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["###", "---", "THE END", "Chapter", "CHAPTER"],
                echo=False
            )
            
            segment_text = output['choices'][0]['text'].strip()
            
            # Clean up any formatting issues
            segment_text = self._clean_text(segment_text)
            
            word_count = len(segment_text.split())
            print(f"Segment {segment_num} generated: {word_count} words")
            
            return segment_text
            
        except Exception as e:
            print(f"Error generating segment {segment_num}: {e}")
            return f"[Error generating segment {segment_num}]"

    def _clean_text(self, text: str) -> str:
        """Clean and format the generated text"""
        # Remove any unwanted formatting
        text = text.replace("```", "")
        text = text.replace("***", "")
        
        # Fix paragraph spacing
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Join with proper paragraph breaks
        return '\n\n'.join(cleaned_lines)

    def generate_complete_story(self, story_data: Dict[str, str], target_total_words: int = 10000) -> Dict[str, str]:
        """Generate a complete cohesive horror story"""
        if not self.model_loaded:
            raise ValueError("Model not loaded - cannot generate story")
        
        print(f"=== Generating complete horror story: '{story_data['title']}' ===")
        print(f"Target length: {target_total_words} words")
        
        # Calculate segments needed (aim for 1500 words per segment)
        words_per_segment = 1500
        num_segments = max(6, target_total_words // words_per_segment)
        
        story_segments = []
        total_words = 0
        
        for segment_num in range(1, num_segments + 1):
            # Adjust target words for final segments
            remaining_words = target_total_words - total_words
            if segment_num == num_segments:
                # Final segment - use remaining words
                segment_target = max(500, remaining_words)
            else:
                segment_target = min(words_per_segment, remaining_words)
            
            if remaining_words <= 0:
                break
                
            segment_text = self.generate_story_segment(
                story_data, segment_num, segment_target
            )
            
            if segment_text and not segment_text.startswith("[Error"):
                story_segments.append(segment_text)
                self.story_context += "\n\n" + segment_text
                segment_words = len(segment_text.split())
                total_words += segment_words
                
                print(f"Progress: {total_words}/{target_total_words} words ({total_words/target_total_words*100:.1f}%)")
                
                # Clean up memory
                gc.collect()
            else:
                print(f"Skipping failed segment {segment_num}")
        
        # Join all segments into complete story
        complete_story = "\n\n".join(story_segments)
        
        # Add a proper ending if the story seems incomplete
        if not self._has_proper_ending(complete_story):
            print("Adding conclusion...")
            ending = self._generate_ending(story_data, complete_story[-500:])
            if ending:
                complete_story += "\n\n" + ending
                total_words += len(ending.split())
        
        print(f"=== Story generation complete ===")
        print(f"Final word count: {total_words} words")
        
        return {
            'title': story_data['title'],
            'text': complete_story,
            'word_count': total_words
        }

    def _has_proper_ending(self, text: str) -> bool:
        """Check if the story has a proper conclusion"""
        ending_indicators = [
            'the end', 'finally', 'at last', 'concluded', 'finished',
            'never again', 'forever', 'eternal', 'final', 'closed his eyes',
            'breathed his last', 'was over', 'came to an end'
        ]
        
        last_paragraph = text[-200:].lower()
        return any(indicator in last_paragraph for indicator in ending_indicators)

    def _generate_ending(self, story_data: Dict[str, str], recent_context: str) -> str:
        """Generate a proper ending for the story"""
        prompt = f"""Write a powerful conclusion to this psychological horror story titled "{story_data['title']}".

Recent context:
{recent_context}

Write a climactic ending that:
- Resolves the main conflict
- Provides a satisfying but disturbing conclusion
- Is 200-400 words
- Maintains the horror atmosphere
- Gives closure to the characters' fates

Write the conclusion:"""
        
        try:
            output = self.llm(
                prompt,
                max_tokens=600,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["###", "---"],
                echo=False
            )
            return output['choices'][0]['text'].strip()
        except:
            return ""

def read_markdown_file(file_path: str) -> str:
    """Reads markdown file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return ""

def extract_story_data(markdown_text: str) -> Dict[str, str]:
    """Extract story data from markdown"""
    lines = markdown_text.split('\n')
    story_title = ""
    
    for line in lines:
        if line.startswith('# ') and not story_title:
            story_title = line[2:].strip()
            break
    
    return {
        'title': story_title or 'The Blood Pact',
        'outline': markdown_text.strip()
    }

def save_story(story: Dict[str, str], filename: str = 'outputs/blood_pact_complete.md'):
    """Save complete story to file"""
    os.makedirs('outputs', exist_ok=True)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {story['title']}\n\n")
            f.write(story['text'])
            f.write(f"\n\n---\n*Complete story: {story['word_count']} words*\n")
        
        print(f"Story saved: {filename} ({story['word_count']} words)")
        return filename
    except Exception as e:
        print(f"Error saving story: {e}")
        return None

def generate_audio(text: str, output_file: str = 'outputs/complete_narration.wav'):
    """Generate audio narration"""
    try:
        print("Generating audio narration...")
        os.makedirs('outputs', exist_ok=True)
        
        pipeline = kokoro.KPipeline(lang_code='a')
        audio_segments = []
        
        # Split text into manageable chunks
        sentences = text.split('. ')
        chunk_size = 5  # Process 5 sentences at a time
        
        for i in range(0, len(sentences), chunk_size):
            chunk = '. '.join(sentences[i:i+chunk_size])
            if chunk:
                generator = pipeline(chunk, voice='bm_fable')
                
                for j, (gs, ps, audio) in enumerate(generator):
                    print(f"Processing audio chunk {i//chunk_size + 1}, segment {j}")
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)

        if audio_segments:
            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_file, combined_audio, 24000)
            duration = len(combined_audio) / 24000
            print(f"Audio saved as '{output_file}' (Duration: {duration:.2f} seconds)")
            return output_file
        else:
            print("No audio segments generated")
            return None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def send_to_telegram(bot_token: str, chat_id: str, text_file: str = None, audio_file: str = None):
    """Send files to Telegram bot"""
    if not bot_token or not chat_id:
        print("Telegram credentials not provided")
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    
    files_to_send = []
    if text_file and os.path.exists(text_file):
        files_to_send.append(('story', text_file))
    if audio_file and os.path.exists(audio_file):
        files_to_send.append(('audio', audio_file))
    
    for file_type, file_path in files_to_send:
        try:
            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {'chat_id': chat_id}
                response = requests.post(url, data=data, files=files)
                
                if response.status_code == 200:
                    print(f"Successfully sent {file_type} to Telegram")
                else:
                    print(f"Failed to send {file_type}: {response.status_code}")
        except Exception as e:
            print(f"Error sending {file_type}: {e}")

def main():
    """Main execution function"""
    # Create the outline file
    outline_content = """# The Blood Pact

## Story Outline
In October 2020, Sebenzile Maphanga was brutally murdered in a ritualistic killing orchestrated by her boyfriend, Collen Mathonsi, and a traditional healer (sangoma), Frans Nkuna. Mathonsi, seeking supernatural powers, was advised by Nkuna to sacrifice Maphanga. He lured her to Nkuna's premises, where she was rendered unconscious, bludgeoned to death with a hammer, and her blood drained into a bucket. Her body was then mutilated and disposed of in the Dubai informal settlement, north of Pretoria.

The story follows the psychological horror of obsession, betrayal, and the dark world of blood magic. It explores the twisted relationship between Collen and Sebenzile, the manipulation by the sangoma, and the devastating consequences of seeking power through violence.

The narrative builds tension through the perspectives of multiple characters - the doomed victim who slowly realizes her danger, the conflicted boyfriend descending into madness, and the calculating sangoma who orchestrates the horror. The story climaxes with the brutal ritual and concludes with the aftermath as justice slowly unfolds."""

    # Write outline file
    os.makedirs('outputs', exist_ok=True)
    with open('story_outline.md', 'w', encoding='utf-8') as f:
        f.write(outline_content)
    
    print("=== INITIALIZING HORROR STORY GENERATOR ===")
    generator = OptimizedHorrorGenerator()
    
    if not generator.model_loaded:
        print("ERROR: Model not loaded. Please check your setup.")
        return
    
    # Load story data
    markdown_text = read_markdown_file('story_outline.md')
    story_data = extract_story_data(markdown_text)
    
    print(f"Generating story: '{story_data['title']}'")
    
    # Generate the complete story
    print("\n=== STORY GENERATION ===")
    story = generator.generate_complete_story(story_data, target_total_words=10000)
    
    # Save the story
    story_file = save_story(story)
    
    if story_file:
        print(f"\n=== STORY COMPLETED ===")
        print(f"Title: {story['title']}")
        print(f"Word count: {story['word_count']}")
        print(f"Saved to: {story_file}")
        
        # Generate audio
        print("\n=== AUDIO GENERATION ===")
        audio_file = generate_audio(story['text'])
        
        # Send to Telegram if configured
        print("\n=== TELEGRAM UPLOAD ===")
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        send_to_telegram(bot_token, chat_id, story_file, audio_file)
        
        print("\n=== PROCESS COMPLETE ===")
    else:
        print("ERROR: Failed to save story")

if __name__ == "__main__":
    main()