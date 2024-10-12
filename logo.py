import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from PIL import Image, ImageFilter
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


url = 'https://blog.designcrowd.com/article/744/100-famous-corporate-logos-from-the-top-companies-of-2015'
driver = webdriver.Chrome()
driver.get(url)


def scroll(scrolls, halfpage=False, sleep=0):
    while scrolls > 0:
        if halfpage:
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
        else:
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

        scrolls -= 1
        if sleep > 0:
            time.sleep(sleep)

        print("Scrolls: ", scrolls)


# Function for extracting data
def GetData(url):
    driver.get(url)
    time.sleep(1)
    scroll(1, sleep=3, halfpage=True)

    img_elements = driver.find_elements(By.XPATH, '//*[@id="form1"]/div[3]/div[4]/div[1]/div/div/div[2]/p[19]/strong')
    name_elements = driver.find_elements(By.XPATH, '//*[@id="form1"]/div[3]/div[4]/div[1]/div/div/div[2]/h2[1]')
    img_urls = [img.get_attribute('src') for img in img_elements]
    names = [name.text for name in name_elements]

    return img_urls, names


# Getting image URLs and names
img_urls, names = GetData(url)
# Extracting titles of videos and images
titles = [element.text for element in driver.find_elements(By.XPATH, '//*[@id="form1"]/div[3]/div[4]/div[1]/div/div/div[2]/h2')]
images = [element.get_attribute('src') for element in driver.find_elements(By.XPATH, '//*[@id="form1"]/div[3]/div[4]/div[1]/div/div/div[2]/p/strong/img')]

vid_titles = []
test = 0.78
for title in titles[:100]:
    vid_titles.append(title)
    print(title)
print(len(vid_titles))

image_titles = []
for image in images:
    image_titles.append(image)
    print(image)
print(len(image_titles))

image_folder = "downloaded_images"
os.makedirs(image_folder, exist_ok=True)
for img_url, name in zip(image_titles, vid_titles):
        response = requests.get(img_url)
        image_path = os.path.join(image_folder, f"{name}.jpg")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Image '{name}' saved successfully.")

df = pd.DataFrame({'Video Titles': vid_titles, 'Image Titles': image_titles})
print(df)
df.to_csv('Brands.csv', index=False)
driver.quit()

#####################################################################################################################
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

#########################################################################################################################
# Extract image names from column 1
image_names = df.iloc[:, 0]

# Use LabelEncoder to encode image names
label_encoder = LabelEncoder()
encoded_image_names = label_encoder.fit_transform(image_names)

# Add the encoded image names to the DataFrame
df['encoded_image_names'] = encoded_image_names

# Display the updated DataFrame
print(df)
###########################################################################################################################


def load_and_preprocess_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0


df['processed_images'] = df['Image Titles'].apply(load_and_preprocess_image)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    np.vstack(df['processed_images'].to_numpy()),  # Convert to NumPy array
    df['encoded_image_names'].to_numpy(),
    test_size=0.2,
    random_state=42
)

# Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the CNN model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nCNN Test Accuracy: {test_acc}')

# Predict on the test set
y_pred_cnn = np.argmax(model.predict(X_test), axis=1)

# Classification report and confusion matrix for CNN
print("\nCNN Classification Report:")
print(classification_report(y_test, y_pred_cnn))

# Confusion Matrix for CNN
print("\nCNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnn))

# Additional Metrics for CNN
precision_cnn = precision_score(y_test, y_pred_cnn, average='weighted')
recall_cnn = recall_score(y_test, y_pred_cnn, average='weighted')
f1_cnn = f1_score(y_test, y_pred_cnn, average='weighted')

print(f"Precision (CNN): {precision_cnn}")
print(f"Recall (CNN): {recall_cnn}")
print(f"F1-Score (CNN): {f1_cnn}")

#########################################################################################################################



# Preprocessing for SVM (flattening the images)
def load_and_preprocess_image_svm(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img).flatten()
    return img_array / 255.0

X_svm = np.vstack(df['Image Titles'].apply(load_and_preprocess_image_svm).to_numpy())

# Split data for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm, df['encoded_image_names'].to_numpy(), test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train_svm)
X_test_svm = scaler.transform(X_test_svm)

# Building an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Training the classifier
svm_classifier.fit(X_train_svm, y_train_svm)

# Predict on the test set with SVM
y_pred_svm = svm_classifier.predict(X_test_svm)

# Accuracy and metrics for SVM
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
precision_svm = precision_score(y_test_svm, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test_svm, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test_svm, y_pred_svm, average='weighted')

print(f'\nSVM Test Accuracy: {accuracy_svm}')
print(f"Precision (SVM): {precision_svm}")
print(f"Recall (SVM): {recall_svm}")
print(f"F1-Score (SVM): {f1_svm}")

# Classification report and confusion matrix for SVM
print("\nSVM Classification Report:")
print(classification_report(y_test_svm, y_pred_svm))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test_svm, y_pred_svm))

########################################################################################################################




# Display the distribution of encoded image names
plt.figure(figsize=(8, 6))
sns.histplot(df['encoded_image_names'], bins=len(df['encoded_image_names'].unique()))
plt.title('Distribution of Encoded Image Names')
plt.xlabel('Encoded Image Names')
plt.ylabel('Count')
plt.show()

# Function to load and display images from URLs
def display_images(urls, titles):
    fig, axes = plt.subplots(1, len(urls), figsize=(12, 4))

    for i, (url, title) in enumerate(zip(urls, titles)):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(title)

    plt.show()


num_images_to_display = 3
sample_indices = df.sample(num_images_to_display, random_state=42).index
sample_image_urls = df.loc[sample_indices, 'Image Titles'].tolist()
sample_titles = df.loc[sample_indices, 'encoded_image_names'].astype(str).tolist()
display_images(sample_image_urls, sample_titles)


resized_folder = 'preprocessed_images'
normalized_folder = 'normalized_images'
smoothed_folder = 'smoothed_images'

# Function to display the top 6 images from a folder
def display_top_images(folder_path, title):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title)

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i in range(6):
        if i < len(image_files):
            image_path = os.path.join(folder_path, image_files[i])
            img = Image.open(image_path)

            plt.subplot(2, 3, i + 1)
            plt.imshow(img)
            plt.title(f'Image {i + 1}')
            plt.axis('off')

    plt.show()

# Displaying top 6 resized images
display_top_images(resized_folder, title='Top 6 Resized Images')

# Displaying top 6 normalized images
display_top_images(normalized_folder, title='Top 6 Normalized Images')

# Displaying top 6 smoothed images
display_top_images(smoothed_folder, title='Top 6 Smoothed Images')

def display_image_histograms(folder_path, title):
    plt.figure(figsize=(18, 8))
    plt.suptitle(f'Histograms of Pixel Intensities - {title}')

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i in range(6):
        if i < len(image_files):
            image_path = os.path.join(folder_path, image_files[i])
            img = Image.open(image_path)

            plt.subplot(3, 6, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Image {i + 1}')
            plt.axis('off')

            plt.subplot(3, 6, i + 7)
            img_array = np.array(img).flatten()
            plt.hist(img_array, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
            plt.title(f'Hist {i + 1}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')

            plt.subplot(3, 6, i + 13)
            plt.hist(img_array, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75, cumulative=True)
            plt.title(f'CDF {i + 1}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Cumulative Probability')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
display_image_histograms(resized_folder, title='Resized Images')
display_image_histograms(normalized_folder, title='Normalized Images')
display_image_histograms(smoothed_folder, title='Smoothed Images')

#Heatmap
def display_heatmap(image_folder, title, num_images=6):
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Heatmap - {title}')

    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    num_images = min(num_images, len(image_files))

    for i in range(1, 7):
        image_path = os.path.join(image_folder, image_files[i - 1])
        img = Image.open(image_path)

        # Converting to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # Subplot for Image
        plt.subplot(2, num_images, i)
        plt.imshow(img_array, cmap='gray')
        plt.title(f'Image {i}')
        plt.axis('off')

        # Subplot for Heatmap
        plt.subplot(2, num_images, i + num_images)
        sns.heatmap(img_array, cmap='viridis', cbar_kws={'label': 'Pixel Intensity'})
        plt.title(f'Heatmap {i}')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

image_folder = 'preprocessed_images'
display_heatmap(image_folder, title='Processed Image Heatmap')