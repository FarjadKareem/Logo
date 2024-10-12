import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import os

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
