import pandas as pd
import os
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

def load_train_images_from_csv(csv_path, image_folder):
    data = pd.read_csv(csv_path)
    labels = data['ClassId'].values
    images = []

    for index, row in data.iterrows():
        relative_path = row["Path"].replace("Train/", "")
        image_path = os.path.join(image_folder, relative_path)

        if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    print(f"Error reading image: {image_path}")
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(e)
        else:
            print(f"Image not found or not a PNG file: {image_path}")
    
    return images, labels

def load_test_images_from_csv(csv_path, image_folder):
    data = pd.read_csv(csv_path)
    labels = data['ClassId'].values
    images = []

    for index, row in data.iterrows():
        relative_path = row["Path"].replace("Test/", "")
        image_path = os.path.join(image_folder, relative_path)

        if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    print(f"Error reading image: {image_path}")
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(e)
        else:
            print(f"Image not found or not a PNG file: {image_path}")
    
    return images, labels    

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform

def split_data(images, labels):
    test_images, val_images, test_labels, val_labels = train_test_split(images, labels, test_size=0.5, random_state=42)
    return test_images, val_images, test_labels, val_labels
