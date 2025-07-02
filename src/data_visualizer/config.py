"""
Configuration file for data visualization module.
"""

import os

# Get the project root directory (2 levels up from the current file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Paths to datasets
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
IMAGES_DIR = os.path.join(DATASETS_DIR, "images")
LABELS_DIR = os.path.join(DATASETS_DIR, "labels")

# Path to output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualize_data")

# Visualization settings
VISUALIZATION_SETTINGS = {
    # General settings
    "dpi": 300,
    "bbox_inches": "tight",
    # Figure sizes
    "default_figsize": (10, 6),
    "large_figsize": (12, 8),
    "small_figsize": (8, 5),
    # Colors
    "default_color": "skyblue",
    "grade_colors": {
        0: "#4575b4",  # Blue
        1: "#91bfdb",  # Light blue
        2: "#ffffbf",  # Yellow
        3: "#fc8d59",  # Orange
        4: "#d73027",  # Red
    },
    # Annotation settings
    "annotation_line_width": 2,
    "annotation_edge_color": "r",
    "annotation_face_color": "none",
    "annotation_text_color": "white",
    "annotation_text_size": 12,
    "annotation_text_bg_color": "red",
    "annotation_text_bg_alpha": 0.7,
    # Histogram settings
    "default_bins": 30,
    "kde": True,
    # Heatmap settings
    "heatmap_cmap": "hot",
    "heatmap_interpolation": "nearest",
    # Box plot settings
    "boxplot_width": 0.5,
    "boxplot_notch": False,
    # Scatter plot settings
    "scatter_alpha": 0.6,
    "scatter_size": 50,
    # Bar chart settings
    "bar_width": 0.8,
    "bar_edge_color": "black",
    "bar_edge_width": 0.5,
    # Pie chart settings
    "pie_shadow": True,
    "pie_startangle": 90,
    "pie_autopct": "%1.1f%%",
    # Line chart settings
    "line_width": 2,
    "line_style": "-",
    "line_marker": "o",
    "line_marker_size": 5,
    # Grid settings
    "grid_alpha": 0.7,
    "grid_linestyle": "--",
    # Text settings
    "title_fontsize": 16,
    "axis_label_fontsize": 12,
    "tick_label_fontsize": 10,
    # Legend settings
    "legend_fontsize": 10,
    "legend_frameon": True,
    "legend_framealpha": 0.8,
    # 3D plot settings
    "3d_elev": 30,
    "3d_azim": 45,
}

# Kellgren-Lawrence specific settings
KL_SETTINGS = {
    "grade_names": {
        0: "Grade 0",
        1: "Grade 1",
        2: "Grade 2",
        3: "Grade 3",
        4: "Grade 4",
    },
    "grade_descriptions": {
        0: "Normal",
        1: "Doubtful narrowing of joint space and possible osteophytic lipping",
        2: "Definite osteophytes, definite narrowing of joint space",
        3: "Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis and possible deformity of bone contour",
        4: "Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone contour",
    },
    "samples_per_grade": 3,  # Number of sample images to show per grade
}

# File extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
LABEL_EXTENSIONS = [".txt"]

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")
