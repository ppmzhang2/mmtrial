import sys
import os

def set_path():
    # Get the directory of this script (set_pythonpath.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the relative path to the custom modules (adjust according to your structure)
    base_dir = os.path.abspath(os.path.join(script_dir, "../src/mmtrial/"))

    # Add the directory to PYTHONPATH (sys.path) if not already added
    if base_dir not in sys.path:
        sys.path.append(base_dir)
