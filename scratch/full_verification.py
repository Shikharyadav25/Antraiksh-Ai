import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageOps
import torch

def run_comprehensive_verification():
    print("==================================================")
    print("   ANTRAIKSHAI SYSTEM INTEGRITY & FORMAT TEST     ")
    print("==================================================")

    # 1. Test Model & Weights Loading
    print("\n[STEP 1] Testing Model Weight & Checkpoint Loading...")
    from model.predict import model, validate_astronomical_image, predict, DEVICE, transform
    print(f"  Device: {DEVICE}")
    print("  Model loaded successfully!")

    # 2. Test Astronomical Validation on PNG, JPG, JPEG & OOD Images
    print("\n[STEP 2] Testing Image Validation & Format Pipeline...")
    base_png_path = "data/test/spiral/0_galaxy_8280.png"
    if not os.path.exists(base_png_path):
        print(f"ERROR: {base_png_path} not found.")
        return

    img_raw = Image.open(base_png_path)
    os.makedirs("scratch/verification_samples", exist_ok=True)

    # Prepare various test formats
    test_cases = {}

    # PNG Format
    png_p = "scratch/verification_samples/sample.png"
    img_raw.save(png_p)
    test_cases["PNG Format"] = (png_p, True)

    # JPG Format
    jpg_p = "scratch/verification_samples/sample.jpg"
    img_raw.save(jpg_p, quality=85)
    test_cases["JPG Format (Quality 85)"] = (jpg_p, True)

    # JPEG Format
    jpeg_p = "scratch/verification_samples/sample.jpeg"
    img_raw.save(jpeg_p, quality=70)
    test_cases["JPEG Format (Quality 70)"] = (jpeg_p, True)

    # Grayscale JPEG
    gray_jpg_p = "scratch/verification_samples/grayscale.jpg"
    img_raw.convert("L").save(gray_jpg_p)
    test_cases["Grayscale JPG"] = (gray_jpg_p, True)

    # EXIF Rotated JPEG
    exif_jpg_p = "scratch/verification_samples/exif.JPG"
    exif = img_raw.getexif()
    exif[0x0112] = 6
    img_raw.save(exif_jpg_p, exif=exif)
    test_cases["EXIF Rotated .JPG"] = (exif_jpg_p, True)

    # Non-Astronomical Daylight Scene (Bright white/gray image)
    daylight_arr = np.ones((160, 160, 3), dtype=np.uint8) * 180
    daylight_img = Image.fromarray(daylight_arr)
    ood_p = "scratch/verification_samples/daylight_ood.jpg"
    daylight_img.save(ood_p)
    test_cases["Non-Astronomical Image (OOD Daylight)"] = (ood_p, False)

    all_passed = True

    for name, (path, expected_is_astro) in test_cases.items():
        print(f"\n---> Testing: {name}")
        img_loaded = Image.open(path)
        img_transposed = ImageOps.exif_transpose(img_loaded).convert("RGB")

        is_astro, dark_ratio, border_mean = validate_astronomical_image(img_transposed)
        print(f"     OOD Result: is_astro={is_astro} (Expected: {expected_is_astro}), dark_ratio={dark_ratio:.3f}, border_mean={border_mean:.1f}")

        # Check prediction output tensor shape
        input_tensor = transform(img_transposed).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        print(f"     Model Output: predicted_idx={pred.item()}, confidence={conf.item()*100:.2f}%")

        if is_astro != expected_is_astro:
            print(f"  [FAILED] expected astronomical status for {name}")
            all_passed = False
        else:
            print(f"  [OK] PASSED {name}")

    # 3. Test App logic & imports
    print("\n[STEP 3] Testing Streamlit App logic & imports...")
    import app
    print("  [OK] app.py imported cleanly without errors!")

    print("\n==================================================")
    if all_passed:
        print("   ALL VERIFICATION TESTS PASSED PERFECTLY!")
    else:
        print("   SOME TESTS FAILED. PLEASE REVIEW LOGS.")
    print("==================================================")

if __name__ == "__main__":
    run_comprehensive_verification()
