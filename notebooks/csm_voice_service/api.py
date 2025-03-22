"""
CSM Voice Service API - Flask API for the CSM Voice Service.

This module provides a REST API for the CSM Voice Service using Flask.
"""

from flask import Flask, request, jsonify
import threading
import base64
from typing import Any, Dict, Optional

from .service import CSMVoiceService


class VoiceServiceAPI:
    """Simple Flask API for the CSM Voice Service.
    
    This class provides a REST API for the CSM Voice Service, allowing
    clients to generate voice audio via HTTP.
    """

    def __init__(self, voice_service: CSMVoiceService, port: int = 5000):
        """Initialize the API server.
        
        Args:
            voice_service: Instance of CSMVoiceService to use
            port: Port to listen on
        """
        self.voice_service = voice_service
        self.port = port
        self.app = Flask("CSM Voice Service API")
        self.setup_routes()
        self._server_thread = None

    def setup_routes(self):
        """Set up the API routes."""

        @self.app.route("/api/generate", methods=["POST"])
        def generate_voice():
            """Generate voice audio from text."""
            data = request.json
            if not data or "text" not in data:
                return jsonify({"error": "Text is required"}), 400

            # Extract parameters with defaults
            text = data["text"]
            speaker_id = data.get("speaker_id", 0)
            max_length_ms = data.get("max_length_ms", 10000)

            # Process context if provided
            context = data.get("context", None)

            try:
                # Generate voice
                result = self.voice_service.generate_voice(
                    text=text,
                    speaker_id=speaker_id,
                    context=context,
                    max_audio_length_ms=max_length_ms
                )

                # Encode audio as base64 if requested
                if data.get("include_audio_data", False):
                    audio_path = result["audio_path"]
                    with open(audio_path, "rb") as audio_file:
                        audio_data = audio_file.read()
                        result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")

                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """Check API health status."""
            return jsonify({
                "status": "ok",
                "device": self.voice_service.device,
                "model_loaded": self.voice_service.generator is not None
            })

    def start(self, debug: bool = False) -> threading.Thread:
        """Start the API server in a separate thread.
        
        Args:
            debug: Whether to run the server in debug mode
            
        Returns:
            Thread: The server thread
        """
        if self._server_thread and self._server_thread.is_alive():
            print(f"Server already running on port {self.port}")
            return self._server_thread
            
        self._server_thread = threading.Thread(
            target=self.app.run,
            kwargs={
                "host": "0.0.0.0",
                "port": self.port,
                "debug": debug,
                "use_reloader": False
            }
        )
        self._server_thread.daemon = True
        self._server_thread.start()
        print(f"API server started on port {self.port}")
        return self._server_thread
        
    def stop(self):
        """Stop the API server if it's running."""
        # Note: Flask doesn't provide a clean way to stop the server
        # This is a limitation of using Flask in a thread
        pass
