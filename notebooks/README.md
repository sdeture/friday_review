# CSM Voice Service

A Python package for generating voice audio using the Conversational Speech Model (CSM) from SesameAI Labs.

## Features

- Text-to-speech generation using CSM 1B model
- Support for conversation context to maintain speaking style
- Persistent storage of generated audio
- REST API for integration with other applications
- Fallback to Edge-TTS when CSM is unavailable

## Installation

```bash
# Clone the CSM repository first
git clone https://github.com/SesameAILabs/csm.git

# Install the package
pip install -e .
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- ffmpeg (for audio processing)
- Hugging Face account with access to the CSM model

## Usage

### Basic usage

```python
from csm_voice_service import CSMVoiceService, setup_environment

# Setup environment (optional)
setup_environment()

# Initialize the service
voice_service = CSMVoiceService()

# Generate voice
result = voice_service.generate_voice(
    text="Hello, I'm a therapeutic assistant. How are you feeling today?",
    speaker_id=0
)

print(f"Generated audio saved to: {result['audio_path']}")
```

### Starting the REST API

```python
from csm_voice_service import CSMVoiceService, VoiceServiceAPI

# Initialize the service
voice_service = CSMVoiceService()

# Initialize and start the API
api = VoiceServiceAPI(voice_service, port=8080)
api.start()

# The API is now running at http://localhost:8080
```

## API Endpoints

### Generate Voice

```
POST /api/generate
```

Request body:
```json
{
    "text": "Your text to convert to speech",
    "speaker_id": 0,
    "max_length_ms": 10000,
    "include_audio_data": true
}
```

### Health Check

```
GET /api/health
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
