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
    llm = Llama.from_pretrained(
        repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
        filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
        n_ctx=8192,
        n_batch=512,
        n_threads=None,
        n_gpu_layers=0,
        verbose=False
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the language model. Error: {e}")
    llm = None

# --- STORY GENERATOR CLASS ---
class KetchumStyleHorrorGenerator:
    """
    Generates horror stories with a six-act structure, each with four sections,
    focusing on crimes against black women in Pretoria, in the style of Jack Ketchum.
    """
    def __init__(self):
        self.llm = llm
        self.model_loaded = llm is not None
        self.section_summaries = []
        self.current_story_data = {}
        self.total_target_words = 7200  # 6 acts * 4 sections * 300 words
        self.health_threshold = 0.75    # Minimum acceptable word count ratio

    def create_story_structure(self, story_idea: Dict[str, str]) -> List[Dict[str, any]]:
        """
        Generates a six-act story structure with four sections per act.
        """
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
                "title": "Investigation",
                "description": "The protagonist digs deeper into the case.",
                "sections": [
                    {"section": 1, "title": "Gathering Clues", "description": f"{protagonist} uncovers evidence linking {crime} to broader issues affecting black women.", "target_words": 300},
                    {"section": 2, "title": "Obstacles", "description": f"Roadblocks emerge, such as community distrust or police corruption in {setting}.", "target_words": 300},
                    {"section": 3, "title": "Breakthrough", "description": f"A key clue or witness points {protagonist} toward {antagonist}.", "target_words": 300},
                    {"section": 4, "title": "Escalation", "description": f"The stakes rise as {antagonist} strikes again, targeting another black woman.", "target_words": 300}
                ]
            },
            {
                "act": 3,
                "title": "Rising Action",
                "description": "The investigation intensifies with greater risks.",
                "sections": [
                    {"section": 1, "title": "Deeper Probe", "description": f"{protagonist} connects {crime} to systemic violence against black women in {setting}.", "target_words": 300},
                    {"section": 2, "title": "Personal Threat", "description": f"{antagonist} targets {protagonist} or someone close, making it personal.", "target_words": 300},
                    {"section": 3, "title": "Increased Danger", "description": f"The danger escalates as {antagonist}’s actions grow bolder.", "target_words": 300},
                    {"section": 4, "title": "Narrowing In", "description": f"{protagonist} closes in on {antagonist}, but key answers remain elusive.", "target_words": 300}
                ]
            },
            {
                "act": 4,
                "title": "Crisis",
                "description": "A major setback challenges the protagonist.",
                "sections": [
                    {"section": 1, "title": "Setback", "description": f"{protagonist} faces a loss—evidence destroyed or a false lead.", "target_words": 300},
                    {"section": 2, "title": "Doubt", "description": f"Doubt creeps in as {protagonist} questions her ability to stop {antagonist}.", "target_words": 300},
                    {"section": 3, "title": "Revelation", "description": f"A shocking truth about {crime} reframes the investigation.", "target_words": 300},
                    {"section": 4, "title": "Regrouping", "description": f"{protagonist} regroups, finding new resolve to face {antagonist}.", "target_words": 300}
                ]
            },
            {
                "act": 5,
                "title": "Climax",
                "description": "The protagonist confronts the antagonist.",
                "sections": [
                    {"section": 1, "title": "Confrontation", "description": f"{protagonist} faces {antagonist} in a tense showdown.", "target_words": 300},
                    {"section": 2, "title": "The Truth", "description": f"The motives behind {crime} and {signature} are fully revealed.", "target_words": 300},
                    {"section": 3, "title": "Struggle", "description": f"A brutal struggle ensues as {protagonist} fights for justice.", "target_words": 300},
                    {"section": 4, "title": "Resolution", "description": f"The conflict ends—{antagonist} is stopped or escapes.", "target_words": 300}
                ]
            },
            {
                "act": 6,
                "title": "Resolution",
                "description": "The aftermath and reflection on the events.",
                "sections": [
                    {"section": 1, "title": "Aftermath", "description": f"The immediate fallout of {crime} affects {protagonist} and {setting}.", "target_words": 300},
                    {"section": 2, "title": "Reflection", "description": f"{protagonist} reflects on the toll of fighting crimes against black women.", "target_words": 300},
                    {"section": 3, "title": "Closure", "description": f"Loose ends tie up, offering {protagonist} some peace.", "target_words": 300},
                    {"section": 4, "title": "Epilogue", "description": f"A glimpse into the future impact on {protagonist} and the community.", "target_words": 300}
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
        """Generates a single section with a detailed prompt."""
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
        print(f"Generating Act {act_num}, Section {section_num}: '{title}'...")
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
                print(f"  ✓ Act {act_num}, Section {section_num} completed ({word_count} words).")
                summary = f"In Act {act_num}, Section {section_num}, {description}"
                self.section_summaries.append(summary)
                return section_text
            else:
                print(f"  ✗ Act {act_num}, Section {section_num} failed quality check.")
                return "[Error: Section generation failed quality validation]"

        except Exception as e:
            print(f"  ✗ Error during generation: {e}")
            return f"[Error: Could not generate Act {act_num}, Section {section_num}]"

    def _clean_text(self, text: str) -> str:
        """Cleans up the model’s output."""
        text = text.strip()
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.match(r'^(WRITING STYLE|INSTRUCTIONS|SECTION|You are|STORY TITLE)', line, re.IGNORECASE)]
        return '\n\n'.join(cleaned_lines).strip()

    def _validate_section(self, section_text: str, target_words: int) -> bool:
        """Ensures the section meets quality standards."""
        if not section_text or section_text.startswith("[Error"):
            return False
        word_count = len(section_text.split())
        if word_count < target_words * self.health_threshold:
            print(f"    - Validation failed: Word count ({word_count}) below threshold.")
            return False
        return True

    def generate_complete_story(self, story_idea: Dict[str, str]) -> Dict[str, any]:
        """Generates a full six-act story with four sections per act."""
        if not self.model_loaded:
            raise ConnectionError("Cannot generate story: model not loaded.")

        print(f"\n=== Generating New Horror Story: '{story_idea['title']}' ===")
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

        print("\n=== Story Generation Complete ===")
        print(f"Final Word Count: {total_words} / {self.total_target_words}")

        return {
            'title': story_idea['title'],
            'text': complete_story_text,
            'word_count': total_words,
        }

# --- UTILITY FUNCTIONS ---
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

def save_story(story: Dict[str, any], output_dir: str = 'outputs') -> None:
    """Saves the story as a Markdown file with act and section headings."""
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r'[^\w\s-]', '', story['title']).strip().replace(' ', '_')
    filename = os.path.join(output_dir, f"{safe_title}.md")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {story['title']}\n\n")
        f.write(f"**Word Count:** {story['word_count']}\n\n")
        f.write(story['text'])
    print(f"Story successfully saved to: {filename}")

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

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    story_prompt = """
# Shadows of Justice
Protagonist: Detective Naledi Mokoena, a fierce black woman seeking justice.
Antagonist: A serial predator targeting black women.
Setting: The gritty streets of Pretoria.
Crime: Abductions and murders of black women.
Signature: A carved symbol left on the victims’ bodies.
"""
    try:
        generator = KetchumStyleHorrorGenerator()
        story_idea = extract_story_idea_from_prompt(story_prompt)
        final_story = generator.generate_complete_story(story_idea)
        if final_story and final_story['word_count'] > 0:
            save_story(final_story)
            # Compute the safe filename outside the f-string to avoid backslash issue
            safe_title = re.sub(r'[^\w\s-]', '', final_story['title']).strip().replace(' ', '_')
            output_file = os.path.join('outputs', f"{safe_title}_Audiobook.wav")
            generate_audio(final_story['text'], output_file=output_file)
        else:
            print("Story generation failed: empty content.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()