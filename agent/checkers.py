import shutil
import subprocess
import platform
import logging
import tempfile
import urllib3 as urllib
import os

logger = logging.getLogger(__name__)


def check_and_install_ollama():
    """
    Check if the 'ollama' CLI is installed, and if not, install it based on the OS:
      - macOS: install via Homebrew.
      - Linux: install via a curl command.
      - Windows: download and run the installer executable.
    """
    if shutil.which("ollama") is not None:
        logger.info("Ollama CLI is already installed.")
        return

    os_type = platform.system()
    logger.info("Ollama CLI not found. Attempting installation on %s...", os_type)

    try:
        if os_type == "Darwin":
            # macOS: install via Homebrew.
            subprocess.check_call(["brew", "install", "ollama"])
        elif os_type == "Linux":
            # Linux: run the install script via curl.
            subprocess.check_call(
                "curl -fsSL https://ollama.com/install.sh | sh", shell=True
            )
        elif os_type == "Windows":
            # Windows: download the installer EXE and run it.
            exe_url = "https://ollama.com/download/OllamaSetup.exe"
            tmp_dir = tempfile.gettempdir()
            exe_path = os.path.join(tmp_dir, "OllamaSetup.exe")
            logger.info("Downloading Ollama installer from %s to %s", exe_url, exe_path)
            urllib.request.urlretrieve(exe_url, exe_path)
            logger.info("Running installer %s", exe_path)
            subprocess.check_call([exe_path])
        else:
            logger.error("Unsupported operating system: %s", os_type)
            raise EnvironmentError("Unsupported operating system.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install Ollama on %s: %s", os_type, e)
        raise EnvironmentError(
            "Ollama installation failed. Please install it manually."
        )
