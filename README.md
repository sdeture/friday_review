# AI-Aligned Therapeutic Assistant

This project combines voice interaction with therapeutic approaches based on Karen Horney's psychoanalytic framework to create an accessible mental health support tool.

## Project Overview

The AI-Aligned Therapeutic Assistant is designed to provide therapeutic interactions through:
- Natural voice communication using advanced speech synthesis
- Therapeutic responses based on Karen Horney's psychoanalytic framework
- Web-based service accessible via Flask API

## Project Structure

- `src/`: Core implementation (framework in development)
  - `therapeutic_framework/`: Karen Horney's therapeutic concepts
  - `utils/`: Security and configuration utilities
  - `voice_provider/`: Voice synthesis integration

- `notebooks/`: Development and demonstration
  - `csm_voice_service.ipynb`: Main implementation of voice service
  - `csm/`: Collaborative Speech Model integration

- `data/`: Reference data for the therapeutic framework
  - `horney_concepts.json`: Structured concepts from Karen Horney's work

## Technology Stack

- **Speech Generation**: CSM (Collaborative Speech Model)
- **Base Model**: Llama 3.2
- **Framework**: PyTorch, Torchaudio
- **Web Interface**: Flask, ngrok
- **Dependencies**: Hugging Face model repository

## Setup and Usage

See the notebook `csm_voice_service.ipynb` for detailed setup and usage instructions.
The system requires GPU acceleration for optimal performance.

## Current Status

This project is in early development with the following components:
- Basic therapeutic framework structure
- Working CSM voice integration
- Initial API service implementation

## Roadmap

- Complete the Karen Horney therapeutic framework implementation
- Improve voice quality and naturalness
- Add session management and history
- Implement progress tracking
