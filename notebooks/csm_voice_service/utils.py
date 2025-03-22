"""
CSM Voice Service Utils - Utility functions and fallback TTS implementation.

This module provides utility functions for the CSM Voice Service, including
a fallback TTS implementation using Edge-TTS when CSM is not available.
"""

import os
import sys
import torch
import torchaudio
import subprocess
import tempfile
import asyncio
from typing import Optional, Any, Tuple


def custom_load_csm_model(device: str = "cuda") -> Any:
    """Custom implementation to load the CSM 1B model properly.
    
    This function handles model configuration and loading more gracefully
    than the default implementation.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Generator: A CSM generator instance
    """
    ensure_csm_imports()
    
    import torch
    import os
    
    try:
        from models import Model, ModelArgs
        from generator import Generator, load_llama3_tokenizer
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
    except ImportError:
        from csm.models import Model, ModelArgs
        from csm.generator import Generator, load_llama3_tokenizer
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

    print("Creating model configuration...")
    # Updated configuration to match the checkpoint dimensions
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,  # Updated from 32000
        audio_vocab_size=2051,   # Updated from 1024
        audio_num_codebooks=32
    )

    print("Initializing model with config...")
    model = Model(config)

    print("Downloading model weights...")
    model_file = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    print(f"Downloaded model weights to {model_file}")

    print("Loading weights into model...")
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)

    print("Moving model to GPU...")
    model = model.to(device=device, dtype=torch.bfloat16)

    print("Setting up model caches...")
    model.setup_caches(1)

    print("Setting up generator...")
    generator = Generator(model)

    print("CSM model loaded successfully!")
    return generator


def ensure_csm_imports() -> None:
    """Ensure CSM imports are available."""
    csm_path = os.path.join(os.getcwd(), 'csm')
    if csm_path not in sys.path and os.path.exists(csm_path):
        sys.path.append(csm_path)
        
    # Also add parent directory in case of nested imports
    parent_path = os.path.dirname(csm_path)
    if parent_path not in sys.path:
        sys.path.append(parent_path)


def create_edge_tts_fallback() -> Any:
    """Create a fallback generator using Edge-TTS.
    
    This function creates a fallback TTS generator that mimics the CSM
    generator API using Microsoft's Edge-TTS.
    
    Returns:
        EdgeTTSGenerator: A generator-like interface using Edge-TTS
    """
    # Install Edge-TTS if needed
    import subprocess, sys
    try:
        import edge_tts
    except ImportError:
        print("Installing Edge-TTS...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
        import edge_tts

    import asyncio
    import os
    import tempfile
    import torch
    import torchaudio

    class EdgeTTSGenerator:
        """A Generator-like interface using Edge-TTS."""

        def __init__(self):
            """Initialize the Edge-TTS generator."""
            self.sample_rate = 24000
            self.voices = {
                0: "en-US-AriaNeural",  # Female voice
                1: "en-US-GuyNeural"    # Male voice
            }
            print("Edge-TTS fallback initialized")

        def generate(self, 
                    text: str, 
                    speaker: int = 0, 
                    context: Optional[list] = None, 
                    max_audio_length_ms: int = 10000) -> torch.Tensor:
            """Generate speech for the given text.
            
            Args:
                text: Text to convert to speech
                speaker: Speaker index (0 or 1)
                context: Optional conversation context (ignored in fallback)
                max_audio_length_ms: Maximum audio length in milliseconds
                
            Returns:
                torch.Tensor: Audio waveform
            """
            voice = self.voices.get(speaker, self.voices[0])

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # Run the Edge-TTS synthesis (requires async)
            async def run_tts():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(temp_path)

            # Run the async function
            asyncio.run(run_tts())

            # Load the audio file
            waveform, sample_rate = torchaudio.load(temp_path)
            waveform = waveform.mean(dim=0)  # Convert to mono

            # Resample if needed
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=self.sample_rate
                )

            # Clean up the temporary file
            os.unlink(temp_path)

            return waveform

    return EdgeTTSGenerator()


def setup_environment() -> None:
    """Setup the environment for CSM.
    
    Sets required environment variables and ensures dependencies.
    """
    # Disable lazy compilation in Mimi as specified in CSM setup
    os.environ["NO_TORCH_COMPILE"] = "1"
    
    # Check for CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: {cuda_available}")
        print(f"CUDA device: {device_name}")
    else:
        print("WARNING: CUDA not available. Performance may be degraded.")
        

def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed.
    
    Returns:
        bool: True if ffmpeg is installed, False otherwise
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def install_ffmpeg() -> bool:
    """Install ffmpeg if not already installed.
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # Different approach depending on platform
        if sys.platform.startswith('linux'):
            subprocess.run(
                ["apt-get", "update", "&&", "apt-get", "install", "-y", "ffmpeg"],
                check=True
            )
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(
                ["brew", "install", "ffmpeg"],
                check=True
            )
        else:
            print("Automatic ffmpeg installation not supported on this platform.")
            print("Please install ffmpeg manually: https://ffmpeg.org/download.html")
            return False
        return True
    except subprocess.SubprocessError:
        print("Failed to install ffmpeg automatically.")
        return False
