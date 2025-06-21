import logging
from pathlib import Path
import re
from typing import Dict, List, Optional
import gc
from llama_cpp import Llama
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class KetchumStyleHorrorGenerator:
    """Generates horror stories in Jack Ketchum's style with a six-act structure."""
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.section_summaries = []
        self.current_story_data = {}
        self.total_target_words = 7200  # Total words for the story
        self.health_threshold = 0.75    # Minimum word count ratio

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        """Defines a six-act story structure with four sections per act."""
        protagonist = story_idea.get('protagonist', 'a determined black female detective')
        antagonist = story_idea.get('antagonist', 'a ruthless killer')
        setting = story_idea.get('setting', 'Pretoria, South Africa')
        crime = story_idea.get('crime', 'a brutal murder of a black woman')
        signature = story_idea.get('signature', 'a chilling mark left at the scene')

        acts = [
            {
                "act": 1, "title": "The Crime", "description": "Introduces the crime and protagonist.",
                "sections": [
                    {"section": 1, "title": "Discovery", "description": f"The story opens with {protagonist} discovering {crime} in {setting}.", "target_words": 300},
                    {"section": 2, "title": "Initial Investigation", "description": f"{protagonist} examines the scene, finding {signature}.", "target_words": 300},
                    {"section": 3, "title": "The Decision", "description": f"{protagonist} resolves to pursue justice.", "target_words": 300},
                    {"section": 4, "title": "First Steps", "description": f"{protagonist} begins investigating in {setting}.", "target_words": 300}
                ]
            },
            {
                "act": 2, "title": "The Hunt Begins", "description": "The protagonist tracks the antagonist.",
                "sections": [
                    {"section": 1, "title": "Gathering Clues", "description": f"{protagonist} uncovers leads about {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "First Encounter", "description": f"A tense near-miss with {antagonist}.", "target_words": 300},
                    {"section": 3, "title": "Allies and Obstacles", "description": f"{protagonist} recruits help but faces resistance.", "target_words": 300},
                    {"section": 4, "title": "A Dark Revelation", "description": f"{protagonist} learns a truth about {signature}.", "target_words": 300}
                ]
            },
            {
                "act": 3, "title": "Descent into Darkness", "description": "The investigation takes a toll.",
                "sections": [
                    {"section": 1, "title": "Personal Cost", "description": f"{protagonist} sacrifices to pursue {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "Another Victim", "description": f"{antagonist} strikes again, leaving {signature}.", "target_words": 300},
                    {"section": 3, "title": "Doubt Creeps In", "description": f"{protagonist} questions her methods.", "target_words": 300},
                    {"section": 4, "title": "A Lead Emerges", "description": f"A breakthrough offers hope.", "target_words": 300}
                ]
            },
            {
                "act": 4, "title": "The Confrontation", "description": "The protagonist closes in.",
                "sections": [
                    {"section": 1, "title": "The Trap", "description": f"{protagonist} sets a plan to catch {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "Ambush", "description": f"A violent clash erupts in {setting}.", "target_words": 300},
                    {"section": 3, "title": "Escape", "description": f"{antagonist} slips away, leaving {signature}.", "target_words": 300},
                    {"section": 4, "title": "Aftermath", "description": f"{protagonist} regroups, more determined.", "target_words": 300}
                ]
            },
            {
                "act": 5, "title": "The Final Pursuit", "description": "The chase reaches its climax.",
                "sections": [
                    {"section": 1, "title": "Cornered", "description": f"{protagonist} tracks {antagonist} to a deadly spot.", "target_words": 300},
                    {"section": 2, "title": "Revelation", "description": f"{antagonist}’s motives tied to {signature} are revealed.", "target_words": 300},
                    {"section": 3, "title": "The Fight", "description": f"A brutal showdown tests {protagonist}.", "target_words": 300},
                    {"section": 4, "title": "Victory or Defeat", "description": f"The outcome hangs in the balance.", "target_words": 300}
                ]
            },
            {
                "act": 6, "title": "Resolution", "description": "The story concludes.",
                "sections": [
                    {"section": 1, "title": "The End", "description": f"{protagonist} faces the consequences.", "target_words": 300},
                    {"section": 2, "title": "Reflection", "description": f"{protagonist} contemplates justice.", "target_words": 300},
                    {"section": 3, "title": "The Mark Remains", "description": f"{signature} lingers as a reminder.", "target_words": 300},
                    {"section": 4, "title": "A New Beginning", "description": f"{protagonist} moves forward, changed.", "target_words": 300}
                ]
            }
        ]
        return acts

    def _build_focused_context(self) -> str:
        """Summarizes the last three sections for context."""
        if not self.section_summaries:
            return ""
        num_to_include = min(3, len(self.section_summaries))
        relevant_summaries = self.section_summaries[-num_to_include:]
        return "PREVIOUSLY IN THE STORY:\n" + "\n".join(relevant_summaries)

    def generate_section(self, act_num: int, section_num: int, section_data: Dict[str, str]) -> str:
        """Generates a single story section."""
        if not self.model_loaded:
            return "[Error: Model not loaded]"

        title = section_data["title"]
        description = section_data["description"]
        target_words = section_data["target_words"]

        ketchum_style_guide = (
            "WRITING STYLE: Emulate Jack Ketchum:\n"
            "- Visceral realism with raw detail.\n"
            "- Human-centered horror, no supernatural elements.\n"
            "- Deep psychological exploration of characters.\n"
            "- Steady tension with brutal honesty.\n"
            "- Authentic Pretoria setting.\n"
            "- Focus on black women’s experiences."
        )

        context = self._build_focused_context()

        prompt = f"""
You are a horror writer like Jack Ketchum, crafting a story about crimes against black women in Pretoria.

STORY TITLE: "{self.current_story_data['title']}"
{context}

CURRENT SECTION: Act {act_num}, Section {section_num} - {title}
SECTION GOAL: {description}

{ketchum_style_guide}

INSTRUCTIONS:
- Write ~{target_words} words.
- Advance the plot based on the goal.
- Focus on new developments and character depth.
- Avoid headers or meta-commentary.
- Begin writing now.
"""

        logger.info(f"Generating Act {act_num}, Section {section_num}: '{title}'...")
        try:
            output = self.llm(
                prompt,
                max_tokens=int(target_words * 1.2),
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.15,
                stop=["\n\n\n", "Section", "Act", "The End"],
                echo=False
            )
            section_text = self._clean_text(output['choices'][0]['text'])

            if self._validate_section(section_text, target_words):
                word_count = len(section_text.split())
                logger.info(f"  ✓ Act {act_num}, Section {section_num} completed ({word_count} words).")
                summary = f"In Act {act_num}, Section {section_num}, {description}"
                self.section_summaries.append(summary)
                return section_text
            else:
                logger.warning(f"  ✗ Act {act_num}, Section {section_num} failed quality check.")
                return "[Error: Section generation failed]"

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return f"[Error: Could not generate Act {act_num}, Section {section_num}]"

    def _clean_text(self, text: str) -> str:
        """Removes unwanted lines from generated text."""
        text = text.strip()
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.match(r'^(WRITING STYLE|INSTRUCTIONS|SECTION|You are|STORY TITLE)', line, re.IGNORECASE)]
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

        story_structure = self.create_story_structure(story_idea)
        full_story_sections = []
        total_words = 0

        for act in story_structure:
            act_num = act['act']
            act_title = act['title']
            full_story_sections.append(f"## Act {act_num}: {act_title}\n\n")
            for section in act['sections']:
                section_num = section['section']
                section_title = section['title']
                section_text = self.generate_section(act_num, section_num, section)
                if not section_text.startswith("[Error"):
                    full_story_sections.append(f"### Section {section_num}: {section_title}\n\n{section_text}\n\n")
                    total_words += len(section_text.split())
                else:
                    full_story_sections.append(f"### Section {section_num}: {section_title}\n\n_{section_text}_\n\n")
                gc.collect()

        complete_story_text = "".join(full_story_sections)

        logger.info("\n=== Story Generation Complete ===")
        logger.info(f"Final Word Count: {total_words} / {self.total_target_words}")

        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words,
        }
def generate_audio(text: str, output_file: str = 'outputs/complete_narration.wav'):
    """Generate audio narration."""
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
            
    except AttributeError as e:
        print(f"Error: Kokoro library issue - {e}. Ensure 'kokoro' is installed and supports KPipeline.")
        return None
    except FileNotFoundError as e:
        print(f"Error: Missing model files - {e}. Ensure required files are available.")
        return None
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def __regenerate_audio(text: str, output_file: str):
    """Generates audio narration with Kokoro TTS and saves it as WAV."""
    logger.info("Generating audio narration...")
    try:
        pipeline = KPipeline(lang_code='a')
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(pipeline(text, voice='en_heart')):
            audio_segments.append(audio)
        if not audio_segments:
            raise ValueError("No audio segments generated.")
        concatenated_audio = np.concatenate(audio_segments)
        sf.write(output_file, concatenated_audio, 24000)
        logger.info(f"Audio saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        raise

def main():
    """Main function to generate story and audio."""
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    story_prompt = """
# Shadows of Pta

Protagonist: Detective Naledi Mokoena, a fierce, deeply empathetic Black woman in her late 30s assigned to Pretoria’s Missing Persons Unit.
Antagonist: A calculating predator who uses charm and deception to lure vulnerable women.
Setting: The divided neighborhoods of greater Pretoria—from well-to-do suburbs to the hidden dangers of tucked-away township streets.
Crime: A string of abductions and murders targeting young Black women. Each victim was last seen on a seemingly innocent “date” after meeting someone through social media or community connections.
Signature: The perpetrator leaves no body—once reported missing, victims vanish without a trace. A single, curated item belonging to each woman (a shoe, an earring) is left conspicuously in their bedroom, along with an ominous carved symbol—an ancient spiral, scratched into a mirror or a wall.

"""

    try:
        logger.info("Extracting story idea...")
        story_idea = extract_story_idea_from_prompt(story_prompt)
        logger.info("Story idea extracted.")

        generator = KetchumStyleHorrorGenerator()
        logger.info("Generating story...")
        final_story = generator.generate_complete_story(story_idea)
        if final_story and final_story['word_count'] > 0:
            story_file = output_dir / f"{final_story['title'].replace(' ', '_')}.md"
            with open(story_file, 'w', encoding='utf-8') as f:
                f.write(f"# {final_story['title']}\n\n")
                f.write(f"**Word Count:** {final_story['word_count']}\n\n")
                f.write(final_story['text'])
            logger.info(f"Story saved to {story_file}")

            audio_file = output_dir / f"{final_story['title'].replace(' ', '_')}_Audiobook.wav"
            generate_audio(final_story['text'], str(audio_file))
        else:
            logger.error("Story generation failed: empty content.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    try:
        generate_audio(text=open('story.md', 'r',encoding='utf-8').read())
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)