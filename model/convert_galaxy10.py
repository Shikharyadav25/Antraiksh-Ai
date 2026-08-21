from pathlib import Path
import h5py
import numpy as np
from PIL import Image

# Bind paths relative to project directory
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "data":
    BASE_DIR = BASE_DIR.parent

DATASET_PATH = BASE_DIR / "data" / "Galaxy10_DECals.h5"
OUTPUT_DIR = BASE_DIR / "data" / "raw"

# Mapping Galaxy10 DECaLS 10-class labels to broad morphologic classes
CLASS_MAP = {
    0: "irregular",   # Distorted Galaxies
    1: "irregular",   # Merging Galaxies
    2: "elliptical",  # Round Smooth Galaxies
    3: "elliptical",  # In-between Round Smooth
    4: "elliptical",  # Cigar Shaped Smooth
    5: "spiral",      # Barred Spiral
    6: "spiral",      # Unbarred Tight Spiral
    7: "spiral",      # Unbarred Loose Spiral
    8: "spiral",      # Edge-on without Bulge
    9: "spiral"       # Edge-on with Bulge
}


def convert():
    """Extracts HDF5 images and organizes them into raw class subdirectories."""
    if not DATASET_PATH.exists():
        print(f"[ERROR] Galaxy10 dataset not found at: {DATASET_PATH}")
        print("Please place 'Galaxy10_DECals.h5' inside the 'data/' folder.")
        return

    # Ensure output folders exist
    for folder in ["spiral", "elliptical", "irregular"]:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    with h5py.File(DATASET_PATH, "r") as file:
        # Access dataset references without loading full arrays into memory
        images_ds = file["images"]
        labels_ds = file["ans"]

        total_images = len(labels_ds)
        print(f"Loaded Galaxy10 dataset containing {total_images} images.")

        counts = {"spiral": 0, "elliptical": 0, "irregular": 0}

        for index in range(total_images):
            label = int(labels_ds[index])
            category = CLASS_MAP.get(label)

            if category is None:
                continue

            # Read image slice dynamically to avoid RAM memory spike
            image_array = np.array(images_ds[index], dtype=np.uint8)

            img = Image.fromarray(image_array)
            if img.mode != "RGB":
                img = img.convert("RGB")

            filename = OUTPUT_DIR / category / f"galaxy_{index}.png"
            img.save(filename)

            counts[category] += 1

            if (index + 1) % 1000 == 0 or (index + 1) == total_images:
                print(f"Processed {index + 1}/{total_images} images...")

    print("\nConversion complete!")
    print("--------------------")
    print(f"Spiral:     {counts['spiral']}")
    print(f"Elliptical: {counts['elliptical']}")
    print(f"Irregular:  {counts['irregular']}")


if __name__ == "__main__":
    convert()