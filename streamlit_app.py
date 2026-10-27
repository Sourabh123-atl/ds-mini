"""COVID/Influenza Trend Analysis Dashboard - Root Entrypoint.
Enables plug-and-play zero-configuration deployment on Streamlit Community Cloud,
Render, Hugging Face Spaces, Railway, and local environments.
"""
from pathlib import Path
import runpy
import sys

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

target_app = ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(target_app), run_name="__main__")
