"""
Setup script for csm_voice_service package.
"""

from setuptools import setup, find_packages

setup(
    name="csm_voice_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "transformers==4.49.0",
        "huggingface_hub==0.28.1",
        "tokenizers==0.21.0",
        "moshi==0.2.2",
        "torchtune==0.4.0",
        "torchao==0.9.0",
        "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master",
        "flask",
        "flask-socketio",
    ],
    extras_require={
        "fallback": ["edge-tts"],
        "dev": ["pytest", "black", "flake8", "isort"],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Voice generation service using the CSM model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/csm_voice_service",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
