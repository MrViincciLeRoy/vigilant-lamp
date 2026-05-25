# vigilant-lamp
## Introduction
This project generates a long-form psychological horror story using a local LLM, converts the story into a WAV audio file for an audiobook, and sends progress updates and the final files via Telegram.
## Key Features
- Automated horror story generation using LLM
- Conversion of story into WAV audio file
- Progress updates and file sharing via Telegram
## Tech Stack
- llama-cpp-python
- numpy
- soundfile
- kokoro-tts
- requests
## Installation
1. Install required libraries: pip install llama-cpp-python numpy soundfile kokoro-tts requests
2. Set Environment Variables: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
3. Ensure you have the GGUF model file available. This script is configured to download it automatically from the Hugging Face Hub.
## Usage
1. Run the horror_generator.py script to start the generation process
2. The script will send progress updates and the final files via Telegram
## Environment Variables
- TELEGRAM_BOT_TOKEN: Your Telegram bot\'s API token
- TELEGRAM_CHAT_ID: The chat ID where notifications and files will be sent
## Code
```\nimport logging\nfrom pathlib import Path\nimport re\nfrom typing import Dict, List\nimport gc\nfrom llama_cpp import Llama\nimport numpy as np\nimport soundfile as sf\nfrom kokoro import KPipeline\nimport os\nimport requests\nimport json\nfrom datetime import datetime\nimport threading\nimport queue\nimport time\n```