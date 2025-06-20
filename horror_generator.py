# -*- coding: utf-8 -*-
"""Optimized Horror Episode Generator - Multi-Chapter Series"""
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
# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="DavidAU/L3-Dark-Planet-8B-GGUF",
	filename="L3-Dark-Planet-8B-D_AU-IQ4_XS.gguf",
)

class OptimizedHorrorGenerator:
    def __init__(self):
        self.llm = llm  # Use the globally defined model
        self.model_loaded = True

    def generate_chapter(self, episode_data: Dict[str, str], chapter_num: int, prev_context: str,
                        max_tokens: int = 2000, temperature: float = 0.8, top_p: float = 0.95,
                        repeat_penalty: float = 1.1) -> str:
        """Generate a single chapter with vivid, emotional storytelling"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        # Preprocess strings to escape double quotes for safe f-string usage
        title = episode_data['title'].replace('"', '""')
        outline = episode_data['outline'].replace('"', '""')
        context_snippet = prev_context[-300:].replace('"', '""')

        # Craft prompt with storytelling instructions
        if chapter_num == 1:
            segment_prompt = f"""# Chapter {chapter_num}: Opening of "{title}"

Write the opening chapter of "{title}" in the visceral, psychological horror style of Jack Ketchum. Create a vivid, emotional journey with:
- 50% dialogue that reveals character motivations and subtext
- 25% narration rich with sensory details (sight, sound, smell, touch, taste)
- 15% body language to show tension and fear
- 10% inner thoughts to deepen emotional impact
Use the outline below to introduce key characters and the setting, building a disturbing atmosphere:

{outline}

Start with a chilling scene that immerses the reader in psychological terror. Focus on imagery, emotional verbs, and sensory details to place the reader in the story. Aim for 1,000–2,000 words.
"""
        else:
            segment_prompt = f"""# Chapter {chapter_num}: Continuation of "{title}"

Continue "{title}" in Jack Ketchum's visceral, psychological horror style. Maintain:
- 50% dialogue with subtext and character voice
- 25% narration with sensory-rich imagery
- 15% body language showing escalating tension
- 10% inner thoughts for emotional depth
Build on the previous chapter:

Previous: ...{context_snippet}

Advance the plot with new horrors, deepening the psychological terror. Use vivid verbs, sensory details, and emotional imagery to sustain a disturbing atmosphere. Aim for 1,000–2,000 words.
"""

        print(f"Generating Chapter {chapter_num}...")
        output = self.llm(
            segment_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["###", "---"],
            echo=False
        )
        chapter_text = output['choices'][0]['text'].strip()
        word_count = len(chapter_text.split())
        print(f"Chapter {chapter_num} generated with {word_count} words")
        return chapter_text

    def generate_series(self, episode_data: Dict[str, str], num_chapters: int = 12,
                       tokens_per_chapter: int = 2000) -> Dict[str, str]:
        """Generate a multi-chapter horror series"""
        chapters = []
        prev_context = ""

        for chapter_num in range(1, num_chapters + 1):
            chapter_text = self.generate_chapter(episode_data, chapter_num, prev_context,
                                                max_tokens=tokens_per_chapter)
            chapters.append(f"# Chapter {chapter_num}\n\n{chapter_text}")
            prev_context = chapter_text
            gc.collect()

        full_text = "\n\n".join(chapters)
        word_count = len(full_text.split())
        print(f"Generated series with {word_count} words across {num_chapters} chapters")
        return {'title': episode_data['title'], 'text': full_text}

def read_markdown_file(file_path: str) -> str:
    """Reads markdown file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_episode_outline(markdown_text: str) -> Dict[str, str]:
    """Extract episode data from markdown"""
    lines = markdown_text.split('\n')
    episode_title = ""
    for line in lines:
        if line.startswith('# ') and not episode_title:
            episode_title = line[2:].strip()
            break
    return {'title': episode_title or 'Untitled Episode', 'outline': markdown_text.strip()}

def save_episode(episode: Dict[str, str], filename: str = 'outputs/blood_pact_series.md'):
    """Save episode series to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {episode['title']}\n\n")
        f.write(episode['text'])
        f.write(f"\n\n---\n*Words: {len(episode['text'].split())}*\n")
    print(f"Series saved: {filename} ({len(episode['text'].split())} words)")

def generate_audio(text: str, output_file: str = 'outputs/complete_narration.wav'):
    """Generate audio narration"""
    pipeline = kokoro.KPipeline(lang_code='a')
    audio_segments = []
    generator = pipeline(text, voice='bm_fable')

    for i, (gs, ps, audio) in enumerate(generator):
        print(f"Processing audio segment {i}: {gs} -> {ps}")
        audio_segments.append(audio)

    if audio_segments:
        combined_audio = np.concatenate(audio_segments, axis=0)
        sf.write(output_file, combined_audio, 24000)
        print(f"Complete narration saved as '{output_file}'")
        print(f"Total duration: {len(combined_audio) / 24000:.2f} seconds")
        return output_file
    else:
        print("No audio segments generated")
        return None

def send_to_telegram(bot_token: str, chat_id: str, text_file: str, audio_file: str):
    """Send files to Telegram bot"""
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    files = [
        ('document', ('blood_pact_series.md', open(text_file, 'rb'))),
        ('document', ('complete_narration.wav', open(audio_file, 'rb')))
    ]
    for file_name, file_data in files:
        data = {'chat_id': chat_id}
        response = requests.post(url, data=data, files={'document': file_data})
        if response.status_code == 200:
            print(f"Successfully sent {file_name} to Telegram")
        else:
            print(f"Failed to send {file_name}: {response.text}")

def main():
    # Create episode outline
    outline_content = """# The Blood Pact
In October 2020, Sebenzile Maphanga was brutally murdered in a ritualistic killing orchestrated by her boyfriend, Collen Mathonsi, and a traditional healer (sangoma), Frans Nkuna. Mathonsi, seeking supernatural powers, was advised by Nkuna to sacrifice Maphanga. He lured her to Nkuna's premises, where she was rendered unconscious, bludgeoned to death with a hammer, and her blood drained into a bucket. Her body was then mutilated and disposed of in the Dubai informal settlement, north of Pretoria. Nkuna led police to the charred remains, and Mathonsi turned state witness, serving a 35-year sentence. Nkuna is currently on trial for murder and kidnapping.

The story closes as the boyfriend turns state witness, confessing everything. Meanwhile, the sangoma awaits trial but claims the ritual was meant for protection, not death. The investigators visit the site where her body was burned, feeling a chill and hearing whispers carried by the wind."""
    
    with open('fast_outline.md', 'w') as f:
        f.write(outline_content)

    # Initialize generator
    generator = OptimizedHorrorGenerator()

    # Load episode data
    markdown_text = read_markdown_file('fast_outline.md')
    episode_data = extract_episode_outline(markdown_text)

    print("=== SERIES GENERATION ===")
    series = generator.generate_series(episode_data, num_chapters=12, tokens_per_chapter=2000)
    save_episode(series)

    print("\n=== AUDIO GENERATION ===")
    audio_file = generate_audio(series['text'])

    print("\n=== SENDING TO TELEGRAM ===")
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if bot_token and chat_id and audio_file:
        send_to_telegram(bot_token, chat_id, 'outputs/blood_pact_series.md', audio_file)
    else:
        print("Telegram bot token or chat ID not set, or audio generation failed")

if __name__ == "__main__":
    main()