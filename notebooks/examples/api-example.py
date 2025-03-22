"""
Example of starting and using the CSM Voice Service API.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csm_voice_service import CSMVoiceService, VoiceServiceAPI, setup_environment
from huggingface_hub import login


def main():
    # Setup environment
    setup_environment()
    
    # Authenticate with Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ")
    
    login(token=hf_token)
    
    # Initialize the voice service
    print("Initializing CSM Voice Service...")
    voice_service = CSMVoiceService()
    
    # Initialize and start the API
    port = 8080
    print(f"Starting API server on port {port}...")
    api = VoiceServiceAPI(voice_service, port=port)
    api_thread = api.start()
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Test the API with a sample request
    test_api(port)
    
    # Keep the server running
    print("\nAPI server is running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def test_api(port):
    """Test the API with a sample request."""
    base_url = f"http://localhost:{port}"
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"Health check response: {response.json()}")
    except Exception as e:
        print(f"Error checking health: {e}")
        return
    
    # Test generate endpoint
    print("\nTesting generate endpoint...")
    test_text = "This is a test of the CSM Voice Service API."
    
    data = {
        "text": test_text,
        "speaker_id": 0,
        "max_length_ms": 5000,
        "include_audio_data": False  # Set to True to include base64 audio data
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/generate", 
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Generation successful!")
            print(f"Audio saved to: {result['audio_path']}")
            print(f"Duration: {result['duration_ms']/1000:.2f} seconds")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error generating speech: {e}")


if __name__ == "__main__":
    main()
