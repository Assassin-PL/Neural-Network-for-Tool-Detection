from linki import ImageLoader
from dataset import ToolDataset, load_images_and_labels
from model import SimpleCNN
from model_training import train_model
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image

def predict(model, image_path):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = (output.data > 0.5).float()
    return prediction.item()

# Example prediction

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {5 * correct / total:.2f}%')

loader = ImageLoader("obrazy/obrazy_linki.txt")
# loader.save_images_to_folder("obrazy")
# images = loader.get_images()
# Base path to your obrazy_do_zmiany
base_path = 'obrazy_do_zmiany'

# Load images and their labels
image_paths, labels = load_images_and_labels(base_path)

# Split the obrazy_do_zmiany into training and testing
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((300, 300)),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets
train_dataset = ToolDataset(train_paths, train_labels, transform=transform)
test_dataset = ToolDataset(test_paths, test_labels, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Train Model and Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, criterion, and optimizer
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, criterion, optimizer, train_loader, epochs=20)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Evaluate the model
evaluate_model(model, test_loader)
# Example prediction
print("czy sa tam miecze??")
print(predict(model, 'obrazy/image_6.jpg'))
