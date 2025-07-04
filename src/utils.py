"""
utils.py – Common Utility Functions
────────────────────────────────────
* Seeding randomness
* Saving matplotlib figures
* Generating simple bar-charts
"""

import os, random, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from datetime import datetime

"""
utils.py – Common Utility Functions
────────────────────────────────────
* Downloading and preparing dataset
* Seeding randomness
* Saving matplotlib figures
* Generating bar charts
* Plotting training curves
"""

import os
import random
import zipfile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ════════════════════════════════════════════════════════
# 1) CORE UTILITIES
# ════════════════════════════════════════════════════════

def seed_everything(seed: int = 42):
    """Fixes randomness for Python, NumPy, TensorFlow, and GPU ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"  # For TensorFlow ≥ 2.12
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _timestamp() -> str:
    """Returns current UTC time as string for filenames."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")

def save_fig(fig, name: str, folder: str = "figures"):
    """
    Saves a matplotlib figure to the specified folder.
    
    Parameters
    ----------
    fig    : Matplotlib figure object
    name   : Filename prefix
    folder : Folder name to save PNG in (default: 'figures')
    """
    proj_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folder_abs = os.path.join(proj_root, folder)
    os.makedirs(folder_abs, exist_ok=True)

    fname = f"{name}_{_timestamp()}.png"
    fpath = os.path.join(folder_abs, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Figure saved ➜ {fpath}")
    return fpath

# ════════════════════════════════════════════════════════
# 2) SIMPLE BAR-CHART
# ════════════════════════════════════════════════════════

def bar_chart(x_labels, series_dict, *, title="", ylabel="# images", save_name=None, folder="figures"):
    """
    Draws grouped bar-charts for comparing multiple data series.

    Parameters
    ----------
    x_labels   : List of x-axis labels
    series_dict: Dictionary of series (label → list of values)
    title      : Chart title (optional)
    ylabel     : Label for the y-axis
    save_name  : If given, saves figure with this name
    folder     : Folder where to save (default: 'figures')
    """
    os.makedirs(folder, exist_ok=True)
    n_series = len(series_dict)
    width = 0.8 / n_series
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (name, y_vals) in enumerate(series_dict.items()):
        offset = (i - n_series / 2) * width + width / 2
        ax.bar(x + offset, y_vals, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    max_y = max(max(vals) for vals in series_dict.values())
    ax.set_ylim(0, max_y * 1.15)

    if n_series > 1:
        ax.legend(title="Split", fontsize=10)

    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, folder)
    plt.show()

# ════════════════════════════════════════════════════════
# 3) TRAINING CURVE PLOTS (Accuracy + Loss)
# ════════════════════════════════════════════════════════

def plot_training_curves(history,
                         *,
                         title_prefix: str = "Model",
                         save_png: bool = True,
                         folder: str = "figures"):
    """
    Plots training history (accuracy and loss) for Keras models.

    Parameters
    ----------
    history      : Keras History object (from .fit())
    title_prefix : Text prefix for titles and saved PNG filename
    save_png     : If True, saves plots to file
    folder       : Output folder for figures (default: 'figures')
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(history.history["accuracy"],     label="Train")
    ax[0].plot(history.history["val_accuracy"], label="Val")
    ax[0].set_title(f"{title_prefix} – Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(history.history["loss"],     label="Train")
    ax[1].plot(history.history["val_loss"], label="Val")
    ax[1].set_title(f"{title_prefix} – Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()

    if save_png:
        png_name = f"{title_prefix.lower().replace(' ', '_')}_curves"
        save_fig(fig, png_name, folder)

    plt.show()

# ════════════════════════════════════════════════════════
# 1) CORE UTILITIES
# ════════════════════════════════════════════════════════
def seed_everything(seed: int = 42):
    """Fixes randomness for Python, NumPy, TF, and (if available) GPU algorithms."""
    os.environ["PYTHONHASHSEED"]      = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"     # TF 2.12+
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _timestamp() -> str:
    """Generate a UTC timestamp for file naming."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def save_fig(fig, name: str, folder: str = "figures"):
    """
    Save a matplotlib figure as PNG into *figures/* folder at project root.
    Returns: full file path.
    """
    # Project root = one level above utils.py
    proj_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folder_abs  = os.path.join(proj_root, folder)
    os.makedirs(folder_abs, exist_ok=True)

    fname = f"{name}_{_timestamp()}.png"
    fpath = os.path.join(folder_abs, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"  Figure saved ➜ {fpath}")
    return fpath


# ════════════════════════════════════════════════════════
# 2) SIMPLE BAR-CHART
# ════════════════════════════════════════════════════════
def bar_chart(x_labels, series_dict, *, title="", ylabel="# images", save_name=None, folder="figures"):
    """
    Draws grouped bar-charts (e.g., for train/val/test).
    Improved style: grid, font, auto-legend control, minimalist design.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    n_series = len(series_dict)
    width = 0.8 / n_series
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (name, y_vals) in enumerate(series_dict.items()):
        offset = (i - n_series/2) * width + width/2
        ax.bar(x + offset, y_vals, width=width, label=name)

    # Style improvements
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    max_y = max(max(vals) for vals in series_dict.values())
    ax.set_ylim(0, max_y * 1.15)

    if n_series > 1:
        ax.legend(title="Split", fontsize=10)

    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, folder)
    plt.show()


# ────────────────────────────────────────────────────────
# 3) TRAINING CURVE PLOTS (Accuracy + Loss)
# ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

def plot_training_curves(history,
                         *,
                         title_prefix: str = "Model",
                         save_png: bool = True,
                         folder: str = "figures"):
    """
    Draws two side-by-side plots:
        • Train / Val Accuracy
        • Train / Val Loss

    Parameters
    ----------
    history       : Keras History object
    title_prefix  : Text prefix for figure titles and filename
    save_png      : If True, saves PNG to the figures/ folder
    folder        : Output directory for the PNG file
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # --- Accuracy ---------------------------------------------------------
    ax[0].plot(history.history["accuracy"],     label="Train")
    ax[0].plot(history.history["val_accuracy"], label="Val")
    ax[0].set_title(f"{title_prefix} – Accuracy")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Acc"); ax[0].legend()

    # --- Loss -------------------------------------------------------------
    ax[1].plot(history.history["loss"],     label="Train")
    ax[1].plot(history.history["val_loss"], label="Val")
    ax[1].set_title(f"{title_prefix} – Loss")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].legend()

    plt.tight_layout()

    # Save as PNG
    if save_png:
        # Convert spaces to underscores → filename
        png_name = f"{title_prefix.lower().replace(' ', '_')}_curves"
        save_fig(fig, png_name, folder)

    plt.show()
