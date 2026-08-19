from pathlib import Path
import h5py
import numpy as np
from PIL import Image

DATASET = Path("data/Galaxy10_DECals.h5")
OUTPUT = Path("data/raw")

CLASS_MAP = {
    0: "irregular",
    1: "irregular",
    2: "elliptical",
    3: "elliptical",
    4: "elliptical",
    5: "spiral",
    6: "spiral",
    7: "spiral",
    8: "spiral",
    9: "spiral"
}


def convert():

    if not DATASET.exists():
        print("Galaxy10_DECals.h5 not found.")
        return

    for folder in ["spiral", "elliptical", "irregular"]:
        (OUTPUT / folder).mkdir(
            parents=True,
            exist_ok=True
        )

    with h5py.File(DATASET, "r") as file:
        images = file["images"][:]
        labels = file["ans"][:]

    print(f"Loaded {len(images)} galaxy images.")

    counts = {
        "spiral": 0,
        "elliptical": 0,
        "irregular": 0
    }

    for index, (image, label) in enumerate(
        zip(images, labels)
    ):

        category = CLASS_MAP.get(int(label))

        if category is None:
            continue

        filename = (
            OUTPUT
            / category
            / f"galaxy_{index}.png"
        )

        Image.fromarray(
            np.asarray(image).astype(np.uint8)
        ).save(filename)

        counts[category] += 1

        if (index + 1) % 500 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(images)}"
            )

    print("\nConversion complete.")
    print(f"Spiral: {counts['spiral']}")
    print(f"Elliptical: {counts['elliptical']}")
    print(f"Irregular: {counts['irregular']}")


if __name__ == "__main__":
    convert()