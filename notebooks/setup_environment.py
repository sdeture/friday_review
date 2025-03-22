#!/usr/bin/env python
"""
Setup script for the CSM Voice Service environment.

This script replaces the csm_venv_setup.ipynb notebook and handles:
1. Creating a virtual environment
2. Installing required packages
3. Cloning the CSM repository
4. Testing imports and dependencies
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, shell=False):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=shell, 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    return result


def create_virtual_environment(venv_name="csm_venv"):
    """Create a Python virtual environment."""
    print(f"\n=== Creating virtual environment: {venv_name} ===")
    
    venv_path = Path(venv_name)
    if venv_path.exists():
        print(f"Virtual environment {venv_name} already exists")
        return venv_path
    
    run_command([sys.executable, "-m", "venv", venv_name])
    return venv_path


def install_packages(venv_path):
    """Install required packages in the virtual environment."""
    print("\n=== Installing required packages ===")
    
    # Determine the correct activation script based on platform
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate"
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Upgrade pip
    run_command(f"source {activate_script} && pip install --upgrade pip", shell=True)
    
    # Install core packages
    run_command(
        f"source {activate_script} && "
        f"pip install torch==2.4.0 torchaudio==2.4.0 transformers==4.49.0 "
        f"huggingface_hub==0.28.1 tokenizers==0.21.0 moshi==0.2.2 torchtune==0.4.0 torchao==0.9.0",
        shell=True
    )
    
    # Install silentcipher
    run_command(
        f"source {activate_script} && "
        f"pip install git+https://github.com/SesameAILabs/silentcipher@master",
        shell=True
    )
    
    # Install Jupyter and ipykernel
    run_command(
        f"source {activate_script} && "
        f"pip install jupyter ipykernel",
        shell=True
    )
    
    # Install API dependencies
    run_command(
        f"source {activate_script} && "
        f"pip install flask flask-socketio",
        shell=True
    )
    
    # Register the kernel
    run_command(
        f"source {activate_script} && "
        f"python -m ipykernel install --user --name={venv_path.name} "
        f"--display-name='Python (CSM Environment)'",
        shell=True
    )
    
    return python_path


def clone_csm_repository():
    """Clone the CSM repository if it doesn't exist yet."""
    print("\n=== Setting up CSM repository ===")
    
    csm_path = Path("csm")
    if csm_path.exists():
        print("CSM repository already exists")
    else:
        run_command(["git", "clone", "https://github.com/SesameAILabs/csm.git"])
    
    return csm_path


def create_directories():
    """Create directories for audio and session data."""
    print("\n=== Creating storage directories ===")
    
    # Create directories
    audio_dir = Path("./psychoanalyst-assistant/generated_audio")
    session_dir = Path("./psychoanalyst-assistant/session_data")
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {audio_dir}")
    print(f"Created directory: {session_dir}")


def install_ffmpeg():
    """Install ffmpeg if not already installed."""
    print("\n=== Installing ffmpeg ===")
    
    try:
        # Check if ffmpeg is already installed
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            print("ffmpeg is already installed")
            return True
    except FileNotFoundError:
        pass
    
    # Install ffmpeg
    if sys.platform.startswith("linux"):
        run_command(["apt-get", "update"], shell=True)
        run_command(["apt-get", "install", "-y", "ffmpeg"], shell=True)
    elif sys.platform == "darwin":  # macOS
        run_command(["brew", "install", "ffmpeg"], shell=True)
    else:
        print("Automatic ffmpeg installation not supported on this platform.")
        print("Please install ffmpeg manually: https://ffmpeg.org/download.html")
        return False
    
    # Verify installation
    try:
        run_command(["ffmpeg", "-version"])
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ffmpeg installation failed")
        return False


def test_imports(python_path):
    """Test importing required modules."""
    print("\n=== Testing imports ===")
    
    # Test silentcipher import
    test_script = "import silentcipher; print('silentcipher successfully imported!')"
    run_command([python_path, "-c", test_script])
    
    # Add CSM to path and test imports
    test_script = """
import os
import sys

csm_path = os.path.join(os.getcwd(), 'csm')
if csm_path not in sys.path:
    sys.path.append(csm_path)

try:
    # First try the direct import
    from csm.generator import load_csm_1b, Segment
    print("Successfully imported CSM modules!")
except ImportError as e:
    print(f"Error importing CSM modules: {e}")
    # If that fails, try with modified imports
    print("Trying alternate import approach...")
    try:
        # Add the parent directory to Python path
        sys.path.append(os.path.dirname(csm_path))
        # Try importing with the module name prefix
        from csm.generator import load_csm_1b, Segment
        print("Successfully imported CSM modules with alternate approach!")
    except ImportError as e:
        print(f"All import attempts failed: {e}")
"""
    run_command([python_path, "-c", test_script])


def main():
    """Main function to set up the environment."""
    parser = argparse.ArgumentParser(description="Set up the CSM Voice Service environment.")
    parser.add_argument("--venv-name", default="csm_venv", help="Name of the virtual environment")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    args = parser.parse_args()
    
    print("====================================================")
    print("Setting up CSM Voice Service environment")
    print("====================================================")
    
    if not args.skip_venv:
        venv_path = create_virtual_environment(args.venv_name)
        python_path = install_packages(venv_path)
    else:
        print("Skipping virtual environment creation")
        python_path = sys.executable
    
    # Clone repository and create directories
    csm_path = clone_csm_repository()
    create_directories()
    install_ffmpeg()
    
    # Test imports
    if not args.skip_venv:
        test_imports(python_path)
    
    print("\n====================================================")
    print("Environment setup complete!")
    print("====================================================")
    print("\nNext steps:")
    print("1. Authenticate with Hugging Face:")
    print("   $ export HF_TOKEN=your_token_here")
    print("2. Run the example scripts:")
    print("   $ python examples/basic_usage.py")
    print("   $ python examples/api_server.py")


if __name__ == "__main__":
    main()
