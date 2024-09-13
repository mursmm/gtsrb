import os

# Define the save directory path
documents_folder = os.path.expanduser("~/Documents")
save_dir = os.path.join(documents_folder, "plots")

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# File paths
train_csv = '/home/mmita/Documents/GTRSB/Train.csv'
test_csv = '/home/mmita/Documents/GTRSB/Test.csv'
image_folder_train = '/home/mmita/Documents/Train/'
image_folder_test = '/home/mmita/Documents/GTRSB/Test'
batch_size = 32
numClasses = 43
learning_rate = 0.0001 
num_epochs = 10