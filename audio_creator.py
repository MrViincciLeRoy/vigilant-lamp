import logging
import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_for_audio(text: str) -> str:
    """Prepares the text for the text-to-speech engine."""
    logger.info("Cleaning text for audio narration...")
    # Remove markdown like headers, bold, italics, and separators
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'^\s*(\* \* \*|---)\s*$', '', text, flags=re.MULTILINE)
    # Normalize whitespace to prevent long pauses
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    logger.info("Text cleaning for audio complete.")
    return text


def generate_audio(text: str, title: str, output_file: str) -> str | None:
    """Generates a WAV audio file from text using the Kokoro TTS engine."""
    try:
        logger.info(f"Starting audio generation for: {title}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize Kokoro TTS pipeline
        # 'a' lang_code means automatic language detection
        pipeline = KPipeline(lang_code='a')
        audio_segments = []
        
        # Split text into chunks (e.g., by paragraph) for stable processing
        text_chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]
        total_chunks = len(text_chunks)
        logger.info(f"Processing {total_chunks} text chunks for audio generation.")
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing audio chunk {i + 1}/{total_chunks}")
            
            # Generate audio for the chunk
            generator = pipeline(chunk, voice='bm_fable')  # 'bm_fable' is a good narrative voice
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)
        
        if audio_segments:
            # Concatenate all audio segments and write to a WAV file
            combined_audio = np.concatenate(audio_segments, axis=0)
            sample_rate = 24000  # Kokoro's default sample rate
            sf.write(output_file, combined_audio, sample_rate)
            duration_sec = len(combined_audio) / sample_rate
            logger.info(f"✅ Audio saved to '{output_file}' - Duration: {duration_sec:.2f} seconds")
            return output_file
        else:
            logger.warning("No audio segments were generated. The text might have been empty.")
            return None
    
    except Exception as e:
        logger.error(f"❌ Critical error during audio generation: {e}", exc_info=True)
        return None


def create_audio_from_text(input_text: str, title: str = None) -> str | None:
    """Main function to create audio from text and save it to the output folder."""
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate title if not provided
    if not title:
        # Use first 50 characters of text as title, cleaned up
        title = input_text[:50].strip()
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'\s+', '_', title)
        if not title:
            title = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Clean title for filename
    title_slug = re.sub(r'[^\w\s-]', '', title)
    title_slug = re.sub(r'\s+', '_', title_slug)
    
    # Prepare text for audio
    clean_text = clean_for_audio(input_text)
    
    if not clean_text.strip():
        logger.error("No valid text content found after cleaning.")
        return None
    
    # Generate audio file path
    audio_file_path = str(output_dir / f"{title_slug}_audio.wav")
    
    # Generate the audio
    result = generate_audio(clean_text, title, audio_file_path)
    
    if result:
        logger.info(f"🎉 Audio creation completed successfully: {result}")
        # Print the file path for GitHub Actions to pick up
        print(f"AUDIO_FILE_PATH={result}")
        return result
    else:
        logger.error("❌ Audio creation failed")
        return None


def main():
    """Main function to run the audio creator."""
    
    # Example usage - you can modify this section
    """sample_text = 
    Welcome to this audio demonstration. This is a simple text-to-speech conversion
    using the Kokoro TTS engine. 
    
    The system can handle multiple paragraphs and will process them chunk by chunk
    to create a smooth audio experience.
    
    This audio file will be saved to the output directory and can be used
    as an artifact in GitHub Actions workflows.
    """
    
    # You can also read text from a file:
    with open('text/starboy.txt', 'r', encoding='utf-8') as f:
         sample_text = f.read()
    
    # Or get text from environment variable:
    # sample_text = os.getenv('INPUT_TEXT', sample_text)
    
    title = os.getenv('AUDIO_TITLE', 'Sample Audio')
    
    result = create_audio_from_text(sample_text, title)
    
    if result:
        print(f"Success! Audio file created: {result}")
        return 0
    else:
        print("Failed to create audio file")
        return 1


if __name__ == "__main__":
    exit(main())