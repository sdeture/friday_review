"""
CSM Voice Service - A Python package for text-to-speech generation using CSM.

This package provides a service for generating voice audio from text using the
Conversational Speech Model (CSM) from SesameAI Labs.
"""

__version__ = '0.1.0'

from .service import CSMVoiceService
from .api import VoiceServiceAPI
from .utils import (
    custom_load_csm_model,
    create_edge_tts_fallback,
    setup_environment,
    check_ffmpeg,
    install_ffmpeg
)

# Make key classes and functions available at package level
__all__ = [
    'CSMVoiceService',
    'VoiceServiceAPI',
    'custom_load_csm_model',
    'create_edge_tts_fallback',
    'setup_environment',
    'check_ffmpeg',
    'install_ffmpeg'
]
