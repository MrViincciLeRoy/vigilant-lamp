import logging
from pathlib import Path
import re
from typing import Dict, List, Optional
import gc
from llama_cpp import Llama
from gtts import gTTS

# Set up logging to console for capture by GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the Llama model with optimized parameters for resource constraints
try:
    logger.info("Loading Llama model...")
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=2048,  # Reduced context window
        n_batch=128,  # Reduced batch size
        n_threads=None,
        n_gpu_layers=0,  # Run on CPU
        verbose=False
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def extract_story_idea_from_prompt(prompt_text: str) -> Dict[str, str]:
    """Extracts story elements from a given prompt text."""
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
    """Generates horror stories in the style of Jack Ketchum with a six-act structure."""
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.section_summaries = []
        self.current_story_data = {}
        self.total_target_words = 7200  # Total words for the entire story
        self.health_threshold = 0.75    # Minimum word count ratio for section validation

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        """Defines the six-act story structure with four sections per act."""
        protagonist = story_idea.get('protagonist', 'a determined black female detective')
        antagonist = story_idea.get('antagonist', 'a ruthless killer')
        setting = story_idea.get('setting', 'Pretoria, South Africa')
        crime = story_idea.get('crime', 'a brutal murder of a black woman')
        signature = story_idea.get('signature', 'a chilling mark left at the scene')

        acts = [
            {
                "act": 1,
                "title": "The Crime",
                "description": "Introduces the crime and the protagonist.",
                "sections": [
                    {"section": 1, "title": "Discovery", "description": f"The story opens with {protagonist} discovering {crime} in {setting}, setting a grim tone.", "target_words": 300},
                    {"section": 2, "title": "Initial Investigation", "description": f"{protagonist} examines the scene, finding {signature} that hints at the killer's identity.", "target_words": 300},
                    {"section": 3, "title": "The Decision", "description": f"{protagonist} resolves to pursue justice, driven by personal stakes as a black woman.", "target_words": 300},
                    {"section": 4, "title": "First Steps", "description": f"{protagonist} begins investigating, facing systemic challenges in {setting}.", "target_words": 300}
                ]
            },
            {
                "act": 2,
                "title": "The Hunt Begins",
                "description": "The protagonist starts tracking the antagonist.",
                "sections": [
                    {"section": 1, "title": "Gathering Clues", "description": f"{protagonist} uncovers leads about {antagonist} in {setting}.", "target_words": 300},
                    {"section": 2, "title": "First Encounter", "description": f"A tense near-miss with {antagonist} heightens the stakes.", "target_words": 300},
                    {"section": 3, "title": "Allies and Obstacles", "description": f"{protagonist} recruits help but faces resistance.", "target_words": 300},
                    {"section": 4, "title": "A Dark Revelation", "description": f"{protagonist} learns a disturbing truth about {signature}.", "target_words": 300}
                ]
            },
            {
                "act": 3,
                "title": "Descent into Darkness",
                "description": "The investigation takes a toll.",
                "sections": [
                    {"section": 1, "title": "Personal Cost", "description": f"{protagonist} sacrifices something dear to pursue {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "Another Victim", "description": f"{antagonist} strikes again, leaving {signature}.", "target_words": 300},
                    {"section": 3, "title": "Doubt Creeps In", "description": f"{protagonist} questions her methods in {setting}.", "target_words": 300},
                    {"section": 4, "title": "A Lead Emerges", "description": f"A breakthrough offers hope amidst despair.", "target_words": 300}
                ]
            },
            {
                "act": 4,
                "title": "The Confrontation",
                "description": "The protagonist closes in on the antagonist.",
                "sections": [
                    {"section": 1, "title": "The Trap", "description": f"{protagonist} sets a plan to catch {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "Ambush", "description": f"A violent clash erupts in {setting}.", "target_words": 300},
                    {"section": 3, "title": "Escape", "description": f"{antagonist} slips away, leaving {signature}.", "target_words": 300},
                    {"section": 4, "title": "Aftermath", "description": f"{protagonist} regroups, more determined than ever.", "target_words": 300}
                ]
            },
            {
                "act": 5,
                "title": "The Final Pursuit",
                "description": "The chase reaches its climax.",
                "sections": [
                    {"section": 1, "title": "Cornered", "description": f"{protagonist} tracks {antagonist} to a deadly location in {setting}.", "target_words": 300},
                    {"section": 2, "title": "Revelation", "description": f"{antagonist}’s motives tied to {signature} are revealed.", "target_words": 300},
                    {"section": 3, "title": "The Fight", "description": f"A brutal showdown tests {protagonist}’s limits.", "target_words": 300},
                    {"section": 4, "title": "Victory or Defeat", "description": f"The outcome hangs in the balance.", "target_words": 300}
                ]
            },
            {
                "act": 6,
                "title": "Resolution",
                "description": "The story concludes with lasting impact.",
                "sections": [
                    {"section": 1, "title": "The End", "description": f"{protagonist} faces the consequences of confronting {antagonist}.", "target_words": 300},
                    {"section": 2, "title": "Reflection", "description": f"{protagonist} contemplates justice in {setting}.", "target_words": 300},
                    {"section": 3, "title": "The Mark Remains", "description": f"{signature} lingers as a haunting reminder.", "target_words": 300},
                    {"section": 4, "title": "A New Beginning", "description": f"{protagonist} moves forward, changed forever.", "target_words": 300}
                ]
            }
        ]
        return acts

    def _build_focused_context(self) -> str:
        """Provides summaries of the last three sections for context."""
        if not self.section_summaries:
            return ""
        num_to_include = min(3, len(self.section_summaries))
        relevant_summaries = self.section_summaries[-num_to_include:]
        return "PREVIOUSLY IN THE STORY:\n" + "\n".join(relevant_summaries)

    def generate_section(self, act_num: int, section_num: int, section_data: Dict[str, str]) -> str:
        """Generates a single section of the story."""
        if not self.model_loaded:
            return "[Error: Model not loaded]"

        title = section_data["title"]
        description = section_data["description"]
        target_words = section_data["target_words"]

        ketchum_style_guide = (
            "WRITING STYLE: Emulate Jack Ketchum. This means:\n"
            "- **Visceral Realism:** Depict violence and its effects with raw, unflinching detail.\n"
            "- **Human-Centered Horror:** Focus on human cruelty, not supernatural elements.\n"
            "- **Psychological Depth:** Explore characters’ fears, motivations, and breaking points.\n"
            "- **Unflinching Pace:** Build tension steadily with brutal honesty.\n"
            "- **Authentic Setting:** Root the story in Pretoria’s real streets, tensions, and atmosphere.\n"
            "- **Focus on Black Women:** Authentically portray the experiences of black women in Pretoria, addressing gender-based violence, systemic inequality, and community impact."
        )

        context = self._build_focused_context()

        prompt = f"""
You are a horror writer in the style of Jack Ketchum, crafting a story about crimes against black women in Pretoria.

STORY TITLE: "{self.current_story_data['title']}"
{context}

CURRENT SECTION: Act {act_num}, Section {section_num} - {title}
SECTION GOAL: {description}

{ketchum_style_guide}

INSTRUCTIONS:
- Write approximately {target_words} words.
- Advance the plot based on the 'SECTION GOAL' without repeating past events.
- Focus on new developments, actions, and character depth.
- Avoid headers or meta-commentary in the text.
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
                return "[Error: Section generation failed quality validation]"

        except Exception as e:
            logger.error(f"  ✗ Error during generation: {e}")
            return f"[Error: Could not generate Act {act_num}, Section {section_num}]"

    def _clean_text(self, text: str) -> str:
        """Cleans up the generated text by removing unwanted lines."""
        text = text.strip()
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.match(r'^(WRITING STYLE|INSTRUCTIONS|SECTION|You are|STORY TITLE)', line, re.IGNORECASE)]
        return '\n\n'.join(cleaned_lines).strip()

    def _validate_section(self, section_text: str, target_words: int) -> bool:
        """Validates if the section meets the word count requirement."""
        if not section_text or section_text.startswith("[Error"):
            return False
        word_count = len(section_text.split())
        if word_count < target_words * self.health_threshold:
            logger.warning(f"    - Validation failed: Word count ({word_count}) below threshold.")
            return False
        return True

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        """Generates the complete six-act story."""
        if not self.model_loaded:
            raise ConnectionError("Cannot generate story: model not loaded.")

        logger.info(f"\n=== Generating New Horror Story: '{story_idea['title']}' ===")
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

def generate_audio(text: str, output_file: str):
    """Generates audio narration for the given text using gTTS."""
    logger.info("Generating audio narration...")
    try:
        tts = gTTS(text, lang='en')
        tts.save(output_file)
        logger.info(f"Audio saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        raise

def main():
    """Main function to generate the story and audio."""
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    story_prompt = """
# Shadows of Justice
Protagonist: Detective Naledi Mokoena, a fierce black woman seeking justice.
Antagonist: A serial predator targeting black women.
Setting: The gritty streets of Pretoria.
Crime: Abductions and murders of black women.
Signature: A carved symbol left on the victims’ bodies.
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

            audio_file = output_dir / f"{final_story['title'].replace(' ', '_')}_Audiobook.mp3"
            generate_audio(final_story['text'], str(audio_file))
        else:
            logger.error("Story generation failed: empty content.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)