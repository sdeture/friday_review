#!/usr/bin/env python
"""
Run script for the CSM Voice Service.

This script provides a simple CLI for running the setup and examples.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    """Run the CSM Voice Service."""
    parser = argparse.ArgumentParser(description="Run the CSM Voice Service.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the environment")
    setup_parser.add_argument("--venv-name", default="csm_venv", help="Name of the virtual environment")
    setup_parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    
    # Basic usage command
    basic_parser = subparsers.add_parser("basic", help="Run the basic usage example")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run the API server example")
    api_parser.add_argument("--port", type=int, default=8080, help="Port to run the API server on")
    
    # Interactive example command
    interactive_parser = subparsers.add_parser("interactive", help="Run the interactive example")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        cmd = [sys.executable, "setup_environment.py"]
        if args.venv_name:
            cmd.extend(["--venv-name", args.venv_name])
        if args.skip_venv:
            cmd.append("--skip-venv")
        subprocess.run(cmd)
    
    elif args.command == "basic":
        subprocess.run([sys.executable, "examples/basic_usage.py"])
    
    elif args.command == "api":
        env = os.environ.copy()
        if args.port:
            env["API_PORT"] = str(args.port)
        subprocess.run([sys.executable, "examples/api_server.py"], env=env)
    
    elif args.command == "interactive":
        subprocess.run([
            sys.executable, 
            "-c", 
            "from csm_voice_service import CSMVoiceService; "
            "from huggingface_hub import login; "
            "import os; "
            "token = os.environ.get('HF_TOKEN') or input('Enter your Hugging Face token: '); "
            "login(token=token); "
            "service = CSMVoiceService(); "
            "service.load_model(); "
            "print('CSM Voice Service loaded. You can now use the service interactively.'); "
            "print('Example: result = service.generate_voice(\"Hello, world!\")'); "
            "import code; "
            "code.interact(local=locals())"
        ])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
