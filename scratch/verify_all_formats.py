import sys
import os
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
from model.predict import predict

def test_all_formats():
    print("==================================================")
    print("VERIFYING PREDICT ON PNG, JPG, JPEG FORMATS")
    print("==================================================")

    sample_png = "data/test/spiral/0_galaxy_8280.png"
    if not os.path.exists(sample_png):
        print(f"Sample file {sample_png} not found.")
        return

    img = Image.open(sample_png)

    os.makedirs("scratch/format_test", exist_ok=True)
    png_path = "scratch/format_test/sample.png"
    jpg_path = "scratch/format_test/sample.jpg"
    jpeg_path = "scratch/format_test/sample.jpeg"

    img.save(png_path)
    img.save(jpg_path, quality=90)
    img.save(jpeg_path, quality=90)

    for format_name, file_path in [("PNG", png_path), ("JPG", jpg_path), ("JPEG", jpeg_path)]:
        print(f"\n--- Testing {format_name} ({file_path}) ---")
        predict(file_path)

if __name__ == "__main__":
    test_all_formats()
