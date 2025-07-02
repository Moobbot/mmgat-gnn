"""
Main script to run data visualization.
This script can be called from the root directory to visualize data.
"""

import os
import sys
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import visualization modules
from src.data_visualizer.config import OUTPUT_DIR, IMAGES_DIR, LABELS_DIR



def main():
    """
    Main function to parse arguments and run the appropriate visualization.
    """


if __name__ == "__main__":
    main()
