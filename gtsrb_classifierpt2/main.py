import torch
from torch.utils.data import DataLoader
from data_preprocessing import load_train_images_from_csv, load_test_images_from_csv, get_transforms, split_data
from dataset import TrafficSignDataset
from model import SimpleCNN
from train import train_model
from evaluate import evaluate_model
from config import (train_csv, test_csv, image_folder_train, image_folder_test,
                    numClasses, batch_size, num_epochs, save_dir)
import matplotlib.pyplot as plt   
import time                
import os 

def main():
    
    # Load and preprocess data
    train_images, train_labels = load_train_images_from_csv(train_csv, image_folder_train)
    test_images, test_labels = load_test_images_from_csv(test_csv, image_folder_test)
    test_images, val_images, test_labels, val_labels = split_data(test_images, test_labels)
    
    train_transform, test_transform = get_transforms()

    train_dataset = TrafficSignDataset(images=train_images, labels=train_labels, transform=train_transform)
    val_dataset = TrafficSignDataset(images=val_images, labels=val_labels, transform=test_transform)
    test_dataset = TrafficSignDataset(images=test_images, labels=test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Start the timer
    start_time = time.time()

    # Train the model
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(train_loader, val_loader, device)

    # Evaluate the model
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, device, criterion)


    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"Total training and evaluation time: {elapsed_time:.2f} seconds")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'Custom_CNN_loss_plot.png'))
    plt.show()
    
    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'Custom_CNN_accuracy_plot.png'))
    plt.show()
    

if __name__ == "__main__":
    main()
