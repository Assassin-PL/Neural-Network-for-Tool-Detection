import os
import cv2
from PIL import Image
from torch.utils.data import Dataset

class ToolDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load an image from the file and apply transforms
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (300, 300))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

def load_images_and_labels(base_path):
    image_paths = []
    labels = []

    # Class '1': Pliers (swords)
    for img_name in os.listdir(os.path.join(base_path, 'swords')):
        image_paths.append(os.path.join(base_path, 'swords', img_name))
        labels.append(1)

    # Class '0': Non-Pliers (niekombinerki)
    for img_name in os.listdir(os.path.join(base_path, 'notSwords')):
        image_paths.append(os.path.join(base_path, 'notSwords', img_name))
        labels.append(0)

    return image_paths, labels


