from PIL import Image
import numpy as np
import os

input_folder = 'preprocessed_images'
output_folder = 'normalized_images'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg")):  
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        img_array = np.array(img)

        img_array_normalized = img_array / 255.0

        output_path = os.path.join(output_folder, filename)
        Image.fromarray((img_array_normalized * 255).astype(np.uint8)).save(output_path)

print("Normalization complete. Normalized images saved in:", output_folder)
