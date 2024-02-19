import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Train
train_csv = '/home/mmita/Documents/GTRSB/Train.csv'
train_data = pd.read_csv(train_csv)

train_labels = train_data['ClassId'].values

#image_folder = '/home/mmita/Documents/GTRSB/Train'
image_folder = '/home/mmita/Documents/Train/'
train_images = []

for index, row in train_data.iterrows():
    # Remove "Train" from the path since it's already in image_folder
    relative_path = row["Path"].replace("Train/", "")
    image_path = os.path.join(image_folder, relative_path)
    
    if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                train_images.append(image)
            else:
                print(f"Error reading image: {image_path}")
        except Exception as e:
            print(f"Error processing image: {image_path}")
            print(e)
    else:
        print(f"Image not found or not a PNG file: {image_path}")
        


#Test
test_csv = '/home/mmita/Documents/GTRSB/Test.csv'
test_data = pd.read_csv(test_csv)

test_labels = test_data['ClassId'].values


image_folder = '/home/mmita/Documents/GTRSB/Test'
test_images = []

for index, row in test_data.iterrows():
    # Remove "Train" from the path since it's already in image_folder
    relative_path = row["Path"].replace("Test/", "")
    image_path = os.path.join(image_folder, relative_path)
    
    if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                test_images.append(image)
            else:
                print(f"Error reading image: {image_path}")
        except Exception as e:
            print(f"Error processing image: {image_path}")
            print(e)
    else:
        print(f"Image not found or not a PNG file: {image_path}")
        
        
'''
#Image display

# Assuming train_images is the list containing your images
num_images_to_display = 10

# Get 10 random indices
random_indices = random.sample(range(len(train_images)), num_images_to_display)

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in zip(random_indices, axes.flatten()):
    ax.imshow(train_images[i])
    ax.axis('off')

plt.show()
'''       

class TrafficSignDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        numpy_image = self.images[idx]

        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray((numpy_image * 255).astype('uint8'))

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label
        
        
        
#Split the data into test and validation sets
test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels,test_size=0.5, random_state=42)     


#tTransformation for training data
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    #transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# Transformation for test data
test_transform = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
])

# Transformation for validation data
val_transform = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
])

#Create datasets
train_dataset = TrafficSignDataset(images=train_images, labels=train_labels, transform=train_transform)
test_dataset = TrafficSignDataset(images=test_images, labels=test_labels, transform=test_transform)
val_dataset = TrafficSignDataset(images=val_images, labels=val_labels, transform=val_transform)



#Dataloader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


numClasses = 43


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional convolutional layers
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #  first fully connected layer
        self.fc1_in_features = 256 * 7 * 7  

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_in_features, 512)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)

        return x




        
#Instantiate the model
model = SimpleCNN(num_classes=numClasses)       

#Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define optimizer and criterion function
learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
#adjust learning rate every 5 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)




# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    if len(y_pred.shape) > 1:
        _, y_pred = torch.max(y_pred, 1)

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc
 
# Define early stopping parameters
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = float('inf')
best_model_state = None

    

# Training and evaluation loop
num_epochs = 30
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_acc = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss
        train_acc += accuracy_fn(y_true=labels, y_pred=outputs)
        loss.backward()
        optimizer.step()
     
    # Calculate loss and accuracy per epoch 
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
   

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.inference_mode():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Calculate validation loss
            val_loss += criterion(outputs, labels)     
            
        # Calculate average validation loss           
        val_loss /= len(val_loader)
        accuracy = correct / total   
    
       
    # Check for early stopping
    if val_loss < best_val_loss:
         best_val_loss = val_loss
         early_stopping_counter = 0
         best_model_state = model.state_dict()
    else:
         early_stopping_counter += 1
         if early_stopping_counter >= early_stopping_patience:
             print(f"Early stopping after {epoch} epochs without improvement.")
             break  # Stop training

     
        
        

    # Adjust learning rate
    scheduler.step()

    # Testing
    
    #Initialize lists to store true labels and predicted labels
    true_labels_test = []
    predicted_labels_test = []
    
    model.eval()
    
    #Initialize test_loss and accuracy
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #obtain the predicted class labels
            _, predicted = torch.max(outputs, 1)
            test_loss += criterion(outputs, labels)
            test_acc += accuracy_fn(y_true=labels, y_pred=outputs)
            
            true_labels_test.extend(labels.cpu().numpy())
            predicted_labels_test.extend(predicted.cpu().numpy())
        # Adjust metrics 
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        #calculate precision score on test data
        test_precision = precision_score(true_labels_test, predicted_labels_test, average='weighted', zero_division=1 )
        #calculate recall score on test data
        test_recall = recall_score(true_labels_test, predicted_labels_test,average='weighted')
        #calculate f1 score on test data
        test_f1_score = f1_score(true_labels_test, predicted_labels_test,average='weighted')
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels_test, predicted_labels_test)

  

    print(f"Epoch: {epoch} | Train_loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}% | Val_loss: {val_loss:.2f} | Validation acc: {accuracy*100:.2f}% | Precision: {test_precision*100:.2f}% | Recall Score: {test_recall*100:.2f}% | F1 Score: {test_f1_score*100:.2f}%")



print('Finished Training and Testing')




 




 