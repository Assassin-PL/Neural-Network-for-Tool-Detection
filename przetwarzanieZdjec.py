import cv2
import numpy as np
import os
import random

class ImageProcessorCV:
    def __init__(self, base_dir="obrazy_do_zmiany", target_dir="DataSet"):
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.swords_dir = os.path.join(target_dir, "swords")
        self.not_swords_dir = os.path.join(target_dir, "notSwords")
        self.create_directories()

    def create_directories(self):
        """Tworzy niezbędne katalogi."""
        os.makedirs(self.swords_dir, exist_ok=True)
        os.makedirs(self.not_swords_dir, exist_ok=True)

    def process_images(self):
        """Przetwarza obrazy zgodnie z wymaganiami."""
        self.process_folder("swords", self.swords_dir)
        self.process_folder("niekombinerki", self.not_swords_dir)

    def process_folder(self, source_subfolder, target_subfolder):
        """Przetwarza pojedynczy folder."""
        source_folder = os.path.join(self.base_dir, source_subfolder)
        for image_name in os.listdir(source_folder):
            image_path = os.path.join(source_folder, image_name)
            if os.path.isfile(image_path):
                self.process_and_save_image(image_path, target_subfolder)

    def process_and_save_image(self, image_path, target_subfolder):
        """Przetwarza i zapisuje pojedyncze zdjęcie."""
        # Otwórz i konwertuj do czarno-białego
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Przeskaluj z zachowaniem proporcji
        bw_resized = self.scale_image(img, 300, 300)

        # Generuj i zapisz obroty
        self.save_rotations(bw_resized, image_path, target_subfolder)

    def scale_image(self, img, target_width, target_height):
        """Skaluje obraz z zachowaniem proporcji."""
        original_height, original_width = img.shape

        # Obliczanie nowych rozmiarów z zachowaniem proporcji
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Skaluj obraz
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Utwórz nowy obraz o wymiarach 300x300
        new_img = np.zeros((target_height, target_width), dtype=np.uint8)
        top = (target_height - new_height) // 2
        left = (target_width - new_width) // 2
        new_img[top:top+new_height, left:left+new_width] = img_resized

        return new_img

    def save_rotations(self, img, original_path, target_subfolder):
        """Zapisuje obraz obrócony o losowe kąty."""
        angles = random.sample(range(360), 3)  # Wybór 3 losowych kątów
        base_name = os.path.splitext(os.path.basename(original_path))[0]

        for i, angle in enumerate(angles):
            M = cv2.getRotationMatrix2D((150, 150), angle, 1) # Środek i kąt obrotu
            rotated = cv2.warpAffine(img, M, (300, 300))
            cv2.imwrite(os.path.join(target_subfolder, f"{base_name}_rotated_{i}.jpg"), rotated)

# Użycie klasy
processor = ImageProcessorCV()
processor.process_images()
