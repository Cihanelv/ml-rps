import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from .utils import seed_everything, save_fig, bar_chart

# Constants
CLASSES = ("rock", "paper", "scissors")

"""
eda.py  –  Data exploration, splitting, and preprocessing

Functions:
* raw_overview()       : Quick view of raw data + sample images
* check_raw_sizes()    : Verify dimensions (W,H) of all .png files
* plot_raw_counts()    : Display and save raw data class distribution
* split_data()         : Create stratified train/val/test folders
* plot_split_counts()  : Show and save class counts after split
* show_split_samples() : Display sample images from train set
* show_augmented_samples(): Display augmented samples from training generator
* build_generators()   : Build augmented train/val/test ImageDataGenerators
* generator_stats()    : Print statistics of ImageDataGenerators
"""

# 1) RAW DATA EXPLORATION
# -----------------------------------------------------------------------------
def raw_overview(raw_dir: str = "../data_raw/archive",
                 n_samples: int = 3,
                 fig_dir: str = "figures",
                 verbose: bool = True):
    """
    Display number of images per class and show example images.
    """
    counts = {c: len([f for f in os.listdir(os.path.join(raw_dir, c))
                      if f.endswith(".png")])
              for c in CLASSES}

    if verbose:
        print("RAW DATA OVERVIEW")
        for cls, n in counts.items():
            print(f"  {cls:<8}: {n} images")
        print()

    # Display sample images
    fig, ax = plt.subplots(1, n_samples, figsize=(4 * n_samples, 3))
    for i, cls in enumerate(CLASSES[:n_samples]):
        sample_path = os.path.join(raw_dir, cls,
                                   os.listdir(os.path.join(raw_dir, cls))[0])
        img = Image.open(sample_path)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(cls)
        if verbose and i == 0:
            print(f"Sample image size: {img.size}  (width, height)")

    save_fig(fig, "raw_samples", fig_dir)
    plt.show()

# 1-b) VERIFY RAW IMAGE SIZES
# -----------------------------------------------------------------------------
def check_raw_sizes(raw_dir: str = "../data_raw/archive",
                    verbose: bool = True):
    """
    Check that all .png files have the same dimensions (width, height).
    """
    from collections import Counter
    sizes = []
    for cls in CLASSES:
        for f in os.listdir(os.path.join(raw_dir, cls)):
            if f.endswith(".png"):
                with Image.open(os.path.join(raw_dir, cls, f)) as img:
                    sizes.append(img.size)

    freq = Counter(sizes)
    if len(freq) == 1:
        w, h = next(iter(freq))
        if verbose:
            print(f"All images have the same size: ({w}, {h})\n")
    else:
        if verbose:
            print("Different image sizes detected:")
            for (w, h), cnt in freq.items():
                print(f"  ({w:3d}, {h:3d}) : {cnt} files")
            print()

# 1-c) RAW DATA CLASS DISTRIBUTION
# -----------------------------------------------------------------------------
def plot_raw_counts(raw_dir: str = "../data_raw/archive",
                    fig_dir: str = "figures",
                    verbose: bool = True):
    """
    Display and save bar chart of class distribution in the raw dataset.
    """
    counts = {cls: len([f for f in os.listdir(os.path.join(raw_dir, cls))
                        if f.endswith(".png")])
              for cls in CLASSES}

    if verbose:
        print("RAW DATA COUNTS")
        for cls, n in counts.items():
            print(f"  {cls:<8}: {n} images")
        print()

    bar_chart(
        CLASSES,
        {"raw": [counts[c] for c in CLASSES]},
        title="Distribution in Raw Dataset",
        ylabel="# images",
        save_name="raw_counts",
        folder=fig_dir
    )

# 2) STRATIFIED SPLIT
# -----------------------------------------------------------------------------
def split_data(raw_dir: str = "../data_raw/archive",
               split_dir: str = "../data_split",
               *, seed: int = 42,
               test_ratio: float = 0.20,
               val_ratio: float = 0.20,
               verbose: bool = True):
    """
    Split the rock/paper/scissors dataset into stratified train/val/test folders.
    """
    # Skip splitting if already done
    if os.path.isdir(split_dir) and all(os.path.isdir(os.path.join(split_dir, d))
                                        for d in ("train", "val", "test")):
        if verbose:
            print(f"Directory {split_dir} already exists; skipping split.\n")
        return

    seed_everything(seed)

    # 1) Collect file paths and labels
    files, labels = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        for f in os.listdir(cls_dir):
            if f.endswith(".png"):
                files.append(os.path.join(cls_dir, f))
                labels.append(cls)

    # 2) Stratified split into train+val and test
    train_files, test_files = train_test_split(
        files, test_size=test_ratio, stratify=labels, random_state=seed
    )
    train_labels = [os.path.basename(os.path.dirname(p)) for p in train_files]

    # 3) Further split train into train and validation
    train_files, val_files = train_test_split(
        train_files, test_size=val_ratio, stratify=train_labels, random_state=seed
    )

    # Helper to copy files into split directories
    def _copy(batch, split_name):
        for src in batch:
            cls = os.path.basename(os.path.dirname(src))
            dst_dir = os.path.join(split_dir, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))

    _copy(train_files, "train")
    _copy(val_files, "val")
    _copy(test_files, "test")

    # Summary of split
    if verbose:
        print("Split completed successfully:")
        for name, batch in (("train", train_files), ("val", val_files), ("test", test_files)):
            counts = {c: 0 for c in CLASSES}
            for p in batch:
                counts[os.path.basename(os.path.dirname(p))] += 1
            print(
                f"  {name:<5} total={len(batch):4d}  "
                f"rock={counts['rock']:3d}  paper={counts['paper']:3d}  scissors={counts['scissors']:3d}"
            )
        print()

# 3) PLOT SPLIT COUNTS
# -----------------------------------------------------------------------------
def plot_split_counts(split_dir: str = "../data_split",
                      fig_dir: str = "figures",
                      verbose: bool = True):
    """
    Display and save bar chart of class counts in train/val/test.
    """
    counts = {
        split: [len(os.listdir(os.path.join(split_dir, split, cls))) for cls in CLASSES]
        for split in ("train", "val", "test")
    }

    if verbose:
        print("SPLIT COUNTS")
        for split, arr in counts.items():
            print(f"  {split:<5}", {c: n for c, n in zip(CLASSES, arr)})
        print()

    bar_chart(
        CLASSES,
        counts,
        title="Distribution of Train / Val / Test",
        ylabel="# images",
        save_name="split_counts",
        folder=fig_dir
    )

# 4) SHOW SAMPLE IMAGES FROM SPLIT
# -----------------------------------------------------------------------------
def show_split_samples(split_dir: str = "../data_split",
                       n_per_class: int = 1,
                       fig_dir: str = "figures"):
    """
    Display and save one sample image per class from the training set.
    """
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 3))
    train_dir = os.path.join(split_dir, "train")

    for i, cls in enumerate(CLASSES):
        file_name = next(f for f in os.listdir(os.path.join(train_dir, cls)) if f.endswith(".png"))
        img = Image.open(os.path.join(train_dir, cls, file_name))
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(cls)

    save_fig(fig, "split_samples", fig_dir)
    plt.show()

# Display augmented samples from training generator
# -----------------------------------------------------------------------------
def show_augmented_samples(train_gen, save_path=None, seed=None):
    """
    Take one augmented image per class from the training generator and display them.
    """
    # 1) Recreate iterator with same seed
    datagen = train_gen.image_data_generator
    effective_seed = seed if seed is not None else getattr(train_gen, "seed", None)
    fresh = datagen.flow_from_directory(
        directory=train_gen.directory,
        target_size=train_gen.target_size,
        batch_size=train_gen.batch_size,
        class_mode=train_gen.class_mode,
        shuffle=train_gen.shuffle,
        seed=effective_seed
    )

    # 2) Capture one sample per class
    rev_map = {v: k for k, v in train_gen.class_indices.items()}
    samples = {}
    for X_batch, y_batch in fresh:
        labels = np.argmax(y_batch, axis=1)
        for img_array, lbl in zip(X_batch, labels):
            cls = rev_map[lbl]
            if cls not in samples:
                samples[cls] = img_array
            if len(samples) == len(rev_map):
                break
        if len(samples) == len(rev_map):
            break

    # 3) Plot samples in class index order
    ordered_classes = sorted(samples.keys(), key=lambda c: train_gen.class_indices[c])
    fig, axes = plt.subplots(1, len(ordered_classes), figsize=(12, 4))
    for ax, cls in zip(axes, ordered_classes):
        ax.imshow(samples[cls])
        ax.set_title(cls)
        ax.axis("off")
    plt.tight_layout()

    # 4) Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

# 5) BUILD IMAGE DATA GENERATORS
# -----------------------------------------------------------------------------
def build_generators(split_dir: str = "../data_split",
                     img_size: tuple = (150, 100),
                     batch: int = 32,
                     seed: int = 42):
    """
    Create augmented train, validation, and test ImageDataGenerators.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    seed_everything(seed)

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )

    plain = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )

    val_gen = plain.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = plain.flow_from_directory(
        os.path.join(split_dir, "test"),
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# 6) GENERATOR STATISTICS
# -----------------------------------------------------------------------------
def generator_stats(train_gen, val_gen, test_gen, verbose: bool = True):
    """
    Print class counts and total images for train, validation, and test generators.
    """
    def _count(gen):
        return {cls: int((gen.classes == idx).sum())
                for cls, idx in gen.class_indices.items()}

    train_counts = _count(train_gen)
    val_counts = _count(val_gen)
    test_counts = _count(test_gen)

    if not verbose:
        return train_counts, val_counts, test_counts

    print("GENERATOR CLASS COUNTS")
    print("  Train:", train_counts)
    print("  Val  :", val_counts)
    print("  Test :", test_counts)

    total_counts = {k: train_counts[k] + val_counts[k] + test_counts[k] for k in CLASSES}
    grand_total = sum(total_counts.values())
    print("\nOVERALL COUNTS")
    for split_name, counts in zip(("train", "val", "test"), (train_counts, val_counts, test_counts)):
        split_total = sum(counts.values())
        print(f"  {split_name:<5}: {split_total:4d} images ({100 * split_total / grand_total:4.1f} %)")
    print(f"\nTotal images: {grand_total}\n")
