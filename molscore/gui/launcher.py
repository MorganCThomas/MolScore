import subprocess
import sys
from pathlib import Path

def run_config():
    # Locates config.py relative to this file
    script_path = Path(__file__).parent / "config.py"
    subprocess.run(["streamlit", "run", str(script_path)] + sys.argv[1:])

def run_monitor():
    # Locates monitor.py relative to this file
    script_path = Path(__file__).parent / "monitor.py"
    subprocess.run(["streamlit", "run", str(script_path)] + sys.argv[1:])