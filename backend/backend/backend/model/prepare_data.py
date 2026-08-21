import random
import shutil
from pathlib import Path

# Set random seed for reproducible train/val/test splits
random.seed(42)

# Bind paths relative to project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE = BASE_DIR / "data" / "raw"
DESTINATION = BASE_DIR / "data"

# Supported image formats
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare_dataset():
    """
    Splits raw image folders into Train (70%), Val (15%), and Test (15%) sets.
    """
    if not SOURCE.exists():
        print(f"[ERROR] Source path does not exist: {SOURCE}")
        print("Please create 'data/raw/' and place your class folders inside it.")
        return

    classes = [folder for folder in SOURCE.iterdir() if folder.is_dir()]

    if not classes:
        print("[WARNING] No class folders found inside 'data/raw/'.")
        return

    for class_folder in classes:
        images = [
            img for img in class_folder.rglob("*")
            if img.suffix.lower() in EXTENSIONS
        ]

        if not images:
            print(f"[SKIP] No supported images found in '{class_folder.name}'.")
            continue

        random.shuffle(images)

        total = len(images)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        groups = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split, files in groups.items():
            dest_dir = DESTINATION / split / class_folder.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for idx, img in enumerate(files):
                shutil.copy2(
                    img,
                    dest_dir / f"{idx}_{img.name}"
                )

        print(
            f"Class: {class_folder.name:<12} | "
            f"Train: {len(train_images):<4} | "
            f"Val: {len(val_images):<4} | "
            f"Test: {len(test_images):<4}"
        )


if __name__ == "__main__":
    prepare_dataset()