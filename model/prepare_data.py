from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE = Path("data/raw")
DESTINATION = Path("data")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare_dataset():

    if not SOURCE.exists():
        print("data/raw folder does not exist.")
        return

    classes = [
        folder
        for folder in SOURCE.iterdir()
        if folder.is_dir()
    ]

    if not classes:
        print("No dataset found.")
        print("Put class folders inside data/raw/")
        print("Example:")
        print("data/raw/spiral/")
        print("data/raw/elliptical/")
        print("data/raw/irregular/")
        return

    for class_folder in classes:

        images = [
            image
            for image in class_folder.rglob("*")
            if image.is_file()
            and image.suffix.lower() in EXTENSIONS
        ]

        if not images:
            print(
                f"No images found in {class_folder.name}"
            )
            continue

        random.shuffle(images)

        total = len(images)

        train_end = int(total * 0.70)
        val_end = int(total * 0.85)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        datasets = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split, files in datasets.items():

            destination = (
                DESTINATION
                / split
                / class_folder.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for number, image in enumerate(files):

                new_name = f"{number}_{image.name}"

                shutil.copy2(
                    image,
                    destination / new_name
                )

        print(
            f"{class_folder.name}: "
            f"Train={len(train_images)}, "
            f"Val={len(val_images)}, "
            f"Test={len(test_images)}"
        )


if __name__ == "__main__":
    prepare_dataset()