"""
CSM Voice Service - Core service class for text-to-speech generation.

This module provides the main CSMVoiceService class for generating
voice audio using the CSM (Conversational Speech Model).
"""

import os
import sys
import torch
import torchaudio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Ensure CSM is in the Python path
def ensure_csm_path():
    """Ensure the CSM module is in the Python path."""
    csm_path = os.path.join(os.getcwd(), 'csm')
    if os.path.exists(csm_path) and csm_path not in sys.path:
        sys.path.append(csm_path)
    return csm_path

# Add CSM to path
ensure_csm_path()

# Import CSM modules
try:
    from csm.generator import load_csm_1b, Segment
except ImportError as e:
    print(f"Warning: Could not import CSM modules: {e}")
    print("CSM functionality will not be available until the repository is correctly installed.")


class CSMVoiceService:
    """Wrapper for the CSM voice generation service with persistence.
    
    This class provides methods to generate voice audio from text using the 
    CSM model, with support for conversation context and persistence.
    """

    def __init__(self, storage_dir: Union[str, Path] = './psychoanalyst-assistant'):
        """Initialize the CSM voice service.
        
        Args:
            storage_dir: Directory for storing generated audio and session data
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing CSM Voice Service on {self.device}...")

        self.generator = None
        self.storage_dir = Path(storage_dir)
        self.audio_dir = self.storage_dir / 'generated_audio'
        self.session_dir = self.storage_dir / 'session_data'

        # Ensure directories exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> Any:
        """Load the CSM model.
        
        Returns:
            Generator: The loaded CSM generator model
        """
        if self.generator is None:
            print("Loading CSM 1B model using custom loader...")
            try:
                from .utils import custom_load_csm_model, create_edge_tts_fallback
                self.generator = custom_load_csm_model(device=self.device)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()

                # If CSM fails, try the Edge-TTS fallback
                print("\nFalling back to Edge-TTS...")
                try:
                    self.generator = create_edge_tts_fallback()
                    print("Fallback TTS loaded successfully")
                except Exception as e2:
                    print(f"Error loading fallback TTS: {e2}")
                    traceback.print_exc()
                    raise
        return self.generator

    def generate_voice(self, 
                      text: str, 
                      speaker_id: int = 0, 
                      context: Optional[List[Dict]] = None, 
                      max_audio_length_ms: int = 10000) -> Dict:
        """Generate voice audio from text.

        Args:
            text: The text to convert to speech
            speaker_id: Speaker identifier (0 or 1)
            context: Optional list of previous conversation segments
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            dict: Information about the generated audio including path and metadata
        """
        generator = self.load_model()

        # Process context if provided
        processed_context = []
        if context and isinstance(context, list):
            for segment in context:
                if 'text' in segment and 'speaker' in segment and 'audio_path' in segment:
                    # Load the audio from the path in the segment
                    audio_path = segment['audio_path']
                    if os.path.exists(audio_path):
                        audio_tensor, sample_rate = torchaudio.load(audio_path)
                        audio_tensor = torchaudio.functional.resample(
                            audio_tensor.squeeze(0),
                            orig_freq=sample_rate,
                            new_freq=generator.sample_rate
                        )

                        processed_context.append(
                            Segment(
                                text=segment['text'],
                                speaker=segment['speaker'],
                                audio=audio_tensor
                            )
                        )

        # Generate the audio
        try:
            audio = generator.generate(
                text=text,
                speaker=speaker_id,
                context=processed_context,
                max_audio_length_ms=max_audio_length_ms
            )

            # Save the audio to disk
            audio_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{audio_id}.wav"
            audio_path = self.audio_dir / filename

            torchaudio.save(
                str(audio_path),
                audio.unsqueeze(0).cpu(),
                generator.sample_rate
            )

            # Log the generation
            metadata = {
                'id': audio_id,
                'timestamp': timestamp,
                'text': text,
                'speaker': speaker_id,
                'audio_path': str(audio_path),
                'sample_rate': generator.sample_rate,
                'duration_ms': len(audio) * 1000 / generator.sample_rate
            }

            # Save metadata
            with open(self.session_dir / f"{audio_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return metadata

        except Exception as e:
            print(f"Error generating voice: {e}")
            raise

    def create_conversation_context(self, texts: List[str], speaker_ids: List[int]) -> List[Dict]:
        """Create a conversation context from texts and speaker ids.

        Args:
            texts: List of utterance texts
            speaker_ids: List of speaker identifiers matching texts

        Returns:
            list: Context segments for use in generate_voice
        """
        if len(texts) != len(speaker_ids):
            raise ValueError("Number of texts must match number of speaker IDs")

        context = []

        for i, (text, speaker_id) in enumerate(zip(texts, speaker_ids)):
            # Generate audio for this segment
            metadata = self.generate_voice(
                text=text,
                speaker_id=speaker_id,
                context=context.copy()  # Use the context up to this point
            )

            # Add this segment to the context
            context.append({
                'text': text,
                'speaker': speaker_id,
                'audio_path': metadata['audio_path']
            })

        return context
