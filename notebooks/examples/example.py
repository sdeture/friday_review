"""
Basic usage example for CSM Voice Service.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csm_voice_service import CSMVoiceService, setup_environment, check_ffmpeg, install_ffmpeg
from huggingface_hub import login


def main():
    # Setup the environment
    setup_environment()
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("ffmpeg not found, attempting to install...")
        if install_ffmpeg():
            print("ffmpeg installed successfully.")
        else:
            print("Failed to install ffmpeg. Please install it manually.")
            return
    
    # Authenticate with Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ")
    
    login(token=hf_token)
    
    # Initialize the voice service
    service = CSMVoiceService()
    
    # Test with a simple utterance
    test_text = "Hello, I'm your therapeutic assistant based on Karen Horney's psychoanalytic framework. How are you feeling today?"
    
    print(f"Generating speech for: {test_text}")
    result = service.generate_voice(
        text=test_text,
        speaker_id=0
    )
    
    print(f"Generated audio saved to: {result['audio_path']}")
    print(f"Audio duration: {result['duration_ms'] / 1000:.2f} seconds")
    
    # Create a demo conversation
    create_demo_conversation(service)


def create_demo_conversation(service):
    """Create a demo conversation between assistant and user."""
    print("\nCreating a demo conversation...")
    
    # Define a conversation between assistant (speaker 0) and user (speaker 1)
    texts = [
        "Hello, I'm your therapeutic assistant based on Karen Horney's psychoanalytic framework. How are you feeling today?",  # Assistant
        "I've been feeling anxious lately, especially at work.",  # User
        "I understand. According to Karen Horney's framework, anxiety often stems from basic conflict between opposing forces in our personality. Can you tell me more about when this anxiety appears?",  # Assistant
        "Mostly in meetings when I have to present my ideas to the team.",  # User
        "That suggests what Horney would call a 'moving toward' pattern, where anxiety appears when seeking approval from others. Let's explore this further."  # Assistant
    ]
    
    speaker_ids = [0, 1, 0, 1, 0]  # Alternating between assistant and user
    
    # Create the conversation context
    context = service.create_conversation_context(texts, speaker_ids)
    
    print(f"Created conversation with {len(context)} segments")
    print(f"Last audio saved to: {context[-1]['audio_path']}")


if __name__ == "__main__":
    main()
