from PIL import Image
import numpy as np
import os

folder_path = 'downloaded_images'

output_folder = 'preprocessed_images'
os.makedirs(output_folder, exist_ok=True)


target_size = (256, 256)


for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg")):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize(target_size)

        img_array = np.array(img)

        output_path = os.path.join(output_folder, filename)
        Image.fromarray(img_array).save(output_path)

print("Preprocessing complete. Preprocessed images saved in:", output_folder)
