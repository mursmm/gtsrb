import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler
from config import num_epochs, learning_rate, numClasses
from model import SimpleCNN
from dataset import TrafficSignDataset
from data_preprocessing import get_transforms
from torch.utils.data import DataLoader

def train_model(train_loader, val_loader, device):
    #comment my cnn to use a pretrained
    model = SimpleCNN(num_classes=numClasses)
    '''
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, numClasses) 
    
    model = models.vgg16(weights='IMAGENET1K_V1')
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, numClasses),
    nn.Softmax(dim=1)
)
    '''
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def accuracy_fn(y_true, y_pred):
        if len(y_pred.shape) > 1:
            _, y_pred = torch.max(y_pred, 1)
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_true)) * 100
        return acc

    best_val_loss = float('inf')
    best_model_state = None
    
    '''
    # Early stopping parameters
    early_stopping_patience = 5
    early_stopping_counter = 0
    '''
    best_val_loss = float('inf')
    best_model_state = None
    

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += accuracy_fn(labels, outputs)
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        '''
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc = accuracy_fn(labels, outputs)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        '''
        
        #chat gpt 
        model.eval()
        val_loss = 0
        val_acc = 0
        correct_val_preds = 0  # To accumulate correct predictions
        total_val_samples = 0  # To accumulate total number of samples
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy correctly by accumulating correct predictions
                _, preds = torch.max(outputs, 1)
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)
        
        # Calculate average loss
        val_loss /= len(val_loader)
        
        # Calculate overall accuracy
        val_acc = (correct_val_preds / total_val_samples) * 100
        
        # Store results
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        #till here
        
        '''
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(best_model_state, 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs without improvement.")
                break  # Stop training
       '''         

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies
