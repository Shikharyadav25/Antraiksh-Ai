import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageOps
import torch

from model.predict import predict, validate_astronomical_image

def test_jpeg_issues():
    print("==================================================")
    print("TESTING DETAILED JPEG / JPG EDGE CASES")
    print("==================================================")

    base_png = "data/test/spiral/0_galaxy_8280.png"
    img_orig = Image.open(base_png)

    # 1. Grayscale JPEG (L mode saved directly as JPG)
    img_gray = img_orig.convert("L")
    img_gray.save("scratch/test_grayscale.jpg")

    # 2. JPG with EXIF rotation tag
    exif = img_orig.getexif()
    exif[0x0112] = 6 # 90 degree rotation
    img_orig.save("scratch/test_exif.JPG", exif=exif) # Note uppercase .JPG

    # 3. JPEG with typical web/astrophotography background sky glow (mean intensity ~60-70)
    arr = np.array(img_orig.convert("RGB"), dtype=np.uint8)
    # Simulate sky glow / light pollution common in web JPEGs
    arr_sky_glow = np.clip(arr + 45, 0, 255).astype(np.uint8)
    Image.fromarray(arr_sky_glow).save("scratch/test_sky_glow.jpeg", quality=85)

    # 4. JPEG with border copyright / text / frame
    arr_frame = arr.copy()
    arr_frame[:15, :, :] = 200 # White top bar
    Image.fromarray(arr_frame).save("scratch/test_framed.jpg")

    tests = [
        ("1. Grayscale JPG", "scratch/test_grayscale.jpg"),
        ("2. Uppercase .JPG with EXIF", "scratch/test_exif.JPG"),
        ("3. Web Sky Glow .jpeg", "scratch/test_sky_glow.jpeg"),
        ("4. Framed / Watermarked JPG", "scratch/test_framed.jpg")
    ]

    for title, filepath in tests:
        print(f"\n--- {title} ---")
        img_loaded = Image.open(filepath)
        print(f"Loaded format: {img_loaded.format}, mode: {img_loaded.mode}, size: {img_loaded.size}")

        # Test without ImageOps.exif_transpose vs with
        transposed = ImageOps.exif_transpose(img_loaded)
        print(f"EXIF transposed size: {transposed.size}")

        is_astro, dark_ratio, border_mean = validate_astronomical_image(img_loaded.convert("RGB"))
        print(f"OOD Validation -> is_astro: {is_astro}, dark_ratio: {dark_ratio:.3f}, border_mean: {border_mean:.2f}")

        try:
            predict(filepath)
        except Exception as e:
            print(f"[ERROR during predict]: {e}")

if __name__ == "__main__":
    test_jpeg_issues()
