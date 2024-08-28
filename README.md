# Neural Network for Tool Detection

This project focuses on developing a neural network to detect a specific tool (in this case, a sword) in images by identifying the bounding box around the object. The project involves several stages, including data collection, preprocessing, model training, and evaluation.

## Project Overview

The main objective of this project is to build and train a convolutional neural network (CNN) capable of detecting and localizing a specific tool in an image. This project uses PyTorch for model training and OpenCV for image processing.

## Project Structure

The project is organized as follows:

- **model.py**: Contains the definition of the `SimpleCNN` class, which implements a convolutional neural network for binary classification of images containing the tool or not.
- **model_training.py**: Contains functions for training the neural network, including the training loop and loss calculation.
- **pobieranie.py**: Handles the downloading of images from the web using requests and BeautifulSoup. It allows fetching images based on specific search queries.
- **przetwarzanieZdjec.py**: Contains the `ImageProcessorCV` class for processing images, including resizing, converting to grayscale, and saving rotations of images.
- **dataset.py**: Includes the `ToolDataset` class for managing the dataset and loading images and their corresponding labels for training and testing.
- **linki.py**: Manages loading images from URLs, displaying them, and saving them to a local directory.
- **main.py**: The main script to execute the entire workflow, including data loading, model training, and evaluation.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- PIL (Pillow)
- Requests
- BeautifulSoup
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
2. Install the required dependencies:
    ```bash
    pip install torch torchvision opencv-python pillow requests beautifulsoup4 matplotlib
## Running the Project
1. Download Images: Use the pobieranie.py script to download images of the tool and non-tool objects from the web.
   ```bash
   python pobieranie.py
2. Process Images: Use the przetwarzanieZdjec.py script to preprocess the downloaded images by converting them to grayscale, resizing, and saving rotated versions.
   ```bash
   python przetwarzanieZdjec.py
3. Train the Model: Run the main.py script to load the images, train the CNN model, and evaluate its performance.
   ```bash
   python main.py
4. Evaluate the Model: The trained model is evaluated on a test set to check its accuracy and effectiveness in detecting the tool in images.
   ```bash
   python main.py
5. Make Predictions: Use the trained model to make predictions on new images.
   ```python
   model = SimpleCNN()
   model.load_state_dict(torch.load('model.pth'))
   prediction = predict(model, 'path_to_new_image.jpg')
   print("Prediction:", prediction)
## Model Architecture

The neural network model used in this project is a simple Convolutional Neural Network (CNN) designed for binary classification, specifically to detect whether a specific tool (sword) is present in an image. The model is implemented using PyTorch and consists of several key components, including convolutional layers, activation functions, pooling layers, and fully connected layers.

### Detailed Architecture

1. **Input Layer**:
   - **Input Dimensions**: The input to the model is a grayscale image with dimensions 300x300 pixels. Since the images are grayscale, they have a single channel.
   - **Shape**: `(1, 300, 300)` where `1` is the number of channels (grayscale), and `300x300` is the height and width of the image.

2. **Convolutional Layer 1**:
   - **Number of Filters**: 16
   - **Filter Size**: 3x3
   - **Padding**: 1 (same padding, which keeps the spatial dimensions the same as the input size)
   - **Activation Function**: ReLU (Rectified Linear Unit)
   - **Output Shape**: After applying 16 filters with 3x3 kernels, the output will have the shape `(16, 300, 300)`.

3. **Max Pooling Layer 1**:
   - **Pooling Size**: 2x2
   - **Stride**: 2 (downsamples the input by a factor of 2)
   - **Output Shape**: The output after max pooling will have the shape `(16, 150, 150)`, effectively reducing the spatial dimensions by half.

4. **Convolutional Layer 2**:
   - **Number of Filters**: 32
   - **Filter Size**: 3x3
   - **Padding**: 1 (same padding)
   - **Activation Function**: ReLU
   - **Output Shape**: The output will have the shape `(32, 150, 150)` after applying 32 filters.

5. **Max Pooling Layer 2**:
   - **Pooling Size**: 2x2
   - **Stride**: 2
   - **Output Shape**: After max pooling, the output will have the shape `(32, 75, 75)`, further reducing the spatial dimensions by half.

6. **Flattening Layer**:
   - **Operation**: This layer flattens the 3D tensor into a 1D tensor to prepare it for the fully connected layers.
   - **Output Shape**: The flattened output will have the shape `(32 * 75 * 75) = (180,000)`.

7. **Fully Connected Layer 1**:
   - **Number of Neurons**: 128
   - **Activation Function**: ReLU
   - **Output Shape**: The output from this layer will have the shape `(128)`.

8. **Fully Connected Layer 2 (Output Layer)**:
   - **Number of Neurons**: 1 (since this is a binary classification problem)
   - **Activation Function**: Sigmoid (to output a probability value between 0 and 1)
   - **Output Shape**: The final output shape is `(1)`, representing the probability that the image contains the tool (sword).

### Summary of the Model

- **Convolutional Layers**: These layers are responsible for extracting features from the input image. The first convolutional layer captures basic features like edges, while deeper layers capture more complex patterns.
- **Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps, making the model more computationally efficient and helping to achieve spatial invariance.
- **Fully Connected Layers**: These layers serve as the decision-making part of the network. They combine the features extracted by the convolutional layers and make predictions about the presence of the tool in the image.

### PyTorch Implementation

1. Here is the code implementing the `SimpleCNN` model in PyTorch:

    ```python
      import torch.nn as nn
      
      class SimpleCNN(nn.Module):
          def __init__(self):
              super(SimpleCNN, self).__init__()
              self.conv_layers = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Conv layer with 16 filters
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2),               # Max pooling
                  nn.Conv2d(16, 32, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2)
              )
              self.flatten = nn.Flatten()
              self.fc_layers = nn.Sequential(
                  nn.Linear(32 * 75 * 75, 128),              # Fully connected layer with 128 neurons
                  nn.ReLU(),
                  nn.Linear(128, 1),                         # Output layer with 1 neuron for binary classification
                  nn.Sigmoid()                               # Sigmoid activation for probability output
              )
      
          def forward(self, x):
              x = self.conv_layers(x)
              x = self.flatten(x)
              x = self.fc_layers(x)
              return x

## Data Preprocessing

Data preprocessing is a crucial step in preparing the images for training a neural network. The quality and consistency of the input data can significantly impact the model's performance. In this project, the preprocessing steps involve image loading, conversion, resizing, augmentation, and labeling. These steps ensure that the images are standardized and suitable for input into the neural network.

### Detailed Steps in Data Preprocessing

1. **Image Loading**:
   - **Loading Images**: The images are loaded from the specified directories. There are two main categories:
     - **Tools**: Images containing the specific tool (e.g., swords).
     - **Non-Tools**: Images that do not contain the tool.
   - **Libraries Used**: OpenCV is used for reading the images in grayscale mode to reduce computational complexity and focus on the shape and edges rather than color information.

   ```python
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
  ### Image Conversion
  
  - **Grayscale Conversion**:
    - **Purpose**: Simplifies the data by focusing on shapes and intensity instead of color.
    - **Process**: Images are converted to grayscale using OpenCV, reducing them to a single channel of intensity values.
    - **Benefits**: This reduces computational complexity and highlights edges crucial for object detection.
  
    ```python
    import cv2
  
    # Load the image in grayscale mode
    image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
### Image Resizing

- **Purpose**: Ensures all images have uniform dimensions, making them suitable for batch processing by the neural network.
- **Process**: Images are resized to 300x300 pixels while preserving their aspect ratio to avoid distortion. Padding is added if necessary to maintain the correct dimensions.
- **Benefits**: Standardizing image size allows the model to process images efficiently and consistently.

  ```python
  image = cv2.resize(image, (300, 300))

This section provides a concise explanation of the image resizing step, with the Python code placed at the end.
### Data Augmentation

- **Purpose**: Increases the diversity of the training dataset, improving the model's robustness and generalization.
- **Process**: Images undergo random rotations and are normalized to ensure consistency. This simulates different orientations and lighting conditions.
- **Benefits**: Augmentation helps the model perform better on unseen data by reducing overfitting and making it more invariant to image variations.

  ```python
  from torchvision import transforms

  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Resize((300, 300)),
      transforms.RandomRotation(degrees=45),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

This concise section explains the data augmentation process, with the Python code provided at the end.
### Labeling

- **Purpose**: Assigns each image a label indicating whether it contains the target tool (e.g., sword) or not.
- **Process**: Images are labeled as `1` for containing the tool and `0` for not containing the tool. These labels are used to train the neural network.
- **Benefits**: Proper labeling is crucial for supervised learning, allowing the model to learn the distinction between images with and without the tool.

  ```python
  def load_images_and_labels(base_path):
      image_paths = []
      labels = []

      # Class '1': Tools (e.g., swords)
      for img_name in os.listdir(os.path.join(base_path, 'swords')):
          image_paths.append(os.path.join(base_path, 'swords', img_name))
          labels.append(1)

      # Class '0': Non-Tools (e.g., not swords)
      for img_name in os.listdir(os.path.join(base_path, 'notSwords')):
          image_paths.append(os.path.join(base_path, 'notSwords', img_name))
          labels.append(0)

      return image_paths, labels

This version provides a concise explanation of the labeling process, with the Python code at the end.
### Data Splitting

- **Purpose**: Divides the dataset into training and testing sets to evaluate the model's performance on unseen data.
- **Process**: Typically, 80% of the data is used for training, and 20% is reserved for testing. This ensures the model is trained on most of the data while having a separate set for evaluation.
- **Benefits**: Splitting the data helps in assessing the model's ability to generalize to new, unseen images.

  ```python
  from sklearn.model_selection import train_test_split

  train_paths, test_paths, train_labels, test_labels = train_test_split(
      image_paths, labels, test_size=0.2, random_state=42
  )

This section briefly explains the data splitting process, with the Python code provided at the end.
### DataLoader

- **Purpose**: Facilitates efficient loading, batching, and shuffling of data during model training and evaluation.
- **Process**: The `DataLoader` is set up to handle batches of images, enabling the model to process multiple images simultaneously. It also shuffles the data to ensure that each training epoch sees the data in a different order.
- **Benefits**: Improves training efficiency and model performance by providing well-organized and randomized batches of data.

  ```python
  from torch.utils.data import DataLoader

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

This explanation of the DataLoader setup is concise, with the relevant Python code included at the end.
### Model Training

- **Training**:
  - **Purpose**: Optimizes the model's parameters using binary cross-entropy loss and the Adam optimizer.
  - **Process**: The training loop runs for a set number of epochs, calculating and printing the loss after each epoch.
  
  ```python
  model.train()
  for epoch in range(epochs):
      running_loss = 0.0
      for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item() * images.size(0)
      epoch_loss = running_loss / len(train_loader.dataset)
      print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
### Evaluation

- **Purpose**: Assesses model accuracy on a separate test set.
- **Process**: The model's predictions are compared against the actual labels, and accuracy is calculated.

  ```python
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for images, labels in test_loader:
          outputs = model(images)
          predicted = (outputs.data > 0.5).float()
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  print(f'Accuracy: {100 * correct / total:.2f}%')

This provides a concise explanation of the evaluation process, with the relevant Python code included.
### References

Here are some useful links to help you understand and implement the concepts used in this project:

- [How to Train an Object Detection Model with Keras](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
- [OpenCV Cascade Classifier Training (OpenCV 3.4)](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)
- [OpenCV Cascade Classifier Documentation (OpenCV 4.x)](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

