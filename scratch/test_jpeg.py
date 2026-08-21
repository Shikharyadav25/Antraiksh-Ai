import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch

from model.predict import predict, validate_astronomical_image

def test_jpeg_scenarios():
    print("==================================================")
    print("Testing JPEG/JPG handling in AntraikshAI model")
    print("==================================================")

    base_png = "data/test/spiral/0_galaxy_8280.png"
    if not os.path.exists(base_png):
        print(f"Error: {base_png} not found.")
        return

    img = Image.open(base_png).convert("RGB")

    scenarios = {}

    # Scenario 1: High JPEG Compression (Quality 20)
    img.save("scratch/test_q20.jpg", quality=20)
    scenarios["JPEG Quality 20"] = "scratch/test_q20.jpg"

    # Scenario 2: JPEG Quality 50
    img.save("scratch/test_q50.jpg", quality=50)
    scenarios["JPEG Quality 50"] = "scratch/test_q50.jpg"

    # Scenario 3: JPEG Quality 95
    img.save("scratch/test_q95.jpg", quality=95)
    scenarios["JPEG Quality 95"] = "scratch/test_q95.jpg"

    # Scenario 4: JPEG with slight background brightness boost (e.g. background mean 55 instead of <50)
    img_arr = np.array(img, dtype=np.float32)
    # add small background offset often present in web JPEGs
    img_bright_bg = np.clip(img_arr + 40, 0, 255).astype(np.uint8)
    Image.fromarray(img_bright_bg).save("scratch/test_bright_bg.jpg", quality=85)
    scenarios["JPEG Bright Background (+40)"] = "scratch/test_bright_bg.jpg"

    # Scenario 5: JPEG CMYK mode
    img_cmyk = img.convert("CMYK")
    img_cmyk.save("scratch/test_cmyk.jpg")
    scenarios["JPEG CMYK Color Space"] = "scratch/test_cmyk.jpg"

    # Scenario 6: JPEG with EXIF orientation metadata (e.g. Rotated 90 deg via EXIF)
    exif = img.getexif()
    exif[0x0112] = 6 # 6 means rotated 90 CW in EXIF
    img.save("scratch/test_exif_rot.jpg", exif=exif)
    scenarios["JPEG EXIF Transpose Required"] = "scratch/test_exif_rot.jpg"

    # Scenario 7: JPEG with light border padding / matte (common in downloaded web JPEGs)
    img_bordered = ImageOps.expand(img, border=10, fill='white')
    img_bordered.save("scratch/test_white_border.jpg")
    scenarios["JPEG White Border/Padding"] = "scratch/test_white_border.jpg"

    print(f"\nOriginal PNG ({base_png}):")
    is_astro, dark_ratio, border_mean = validate_astronomical_image(img)
    print(f"  Validation -> is_astro: {is_astro}, dark_ratio: {dark_ratio:.3f}, border_mean: {border_mean:.2f}")

    print("\n--------------------------------------------------")
    for name, path in scenarios.items():
        test_img = Image.open(path)
        is_astro, dark_ratio, border_mean = validate_astronomical_image(test_img)
        print(f"\nScenario: {name}")
        print(f"  File: {path}")
        print(f"  OOD Check -> is_astro: {is_astro}, dark_ratio: {dark_ratio:.3f}, border_mean: {border_mean:.2f}")
        predict(path)

if __name__ == "__main__":
    test_jpeg_scenarios()
