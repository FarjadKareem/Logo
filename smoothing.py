from PIL import Image, ImageFilter
import os

input_folder = 'preprocessed_images'
output_folder = 'smoothed_images'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg")):  # Check if the file is a JPEG image
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        # Applying Gaussian blur for smoothing
        img_smoothed = img.filter(ImageFilter.GaussianBlur(radius=2))  # Adjust the radius as needed

        output_path = os.path.join(output_folder, filename)
        img_smoothed.save(output_path)

print("Smoothing complete. Smoothed images saved in:", output_folder)