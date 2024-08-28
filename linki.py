import cv2
import numpy as np
import urllib.request
from matplotlib import pyplot as plt
import os

class ImageLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.links = self.extract_links_from_file()

    def extract_links_from_file(self):
        # Read links from the provided file
        links = []
        try:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
                links = [line.strip() for line in lines if line.startswith('http')]
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
        return links

    def load_image_from_url(self, url):
        # Load an image from a URL using OpenCV
        try:
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Failed to load image from {url}. Error: {e}")
            return None

    def display_images(self):
        # Display all images loaded from the links
        for i, url in enumerate(self.links):
            image = self.load_image_from_url(url)
            if image is not None:
                plt.figure(figsize=(10, 5))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f"Image {i+1}")
                plt.axis('off')
                plt.show()
            else:
                print(f"Image {i+1} could not be displayed.")

    def save_images_to_folder(self, folder_name):
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save each image to the folder
        for i, url in enumerate(self.links):
            image = self.load_image_from_url(url)
            if image is not None:
                # Define the path for saving the image
                file_extension = 'jpg' if 'jpg' in url.split('.')[-1] else 'png'
                image_path = os.path.join(folder_name, f"image_{i+1}.{file_extension}")
                cv2.imwrite(image_path, image)
                print(f"Saved Image {i+1} as {image_path}")
            else:
                print(f"Image {i+1} could not be saved.")

    def get_images(self):
            images = []
            for url in self.links:
                image = self.load_image_from_url(url)
                if image is not None:
                    images.append(image)
            return images

# Example usage:
# loader = ImageLoader("linki.txt")
# loader.display_images()
