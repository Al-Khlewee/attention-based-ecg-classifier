import torch
import numpy as np
from sklearn.model_selection import train_test_split
# Import from the created modules
from data.preprocessing import load_and_preprocess_data
from data.dataset import ECGDataset
from models.attention_ecg_model import AttentionECGModel
from utils.train_utils import train_model
from utils.visualization import evaluate_model, visualize_attention_enhanced
from config import * # Import configuration parameters
from torch.utils.data import DataLoader  # Import DataLoader
import torch.nn as nn  # Import torch.nn
import matplotlib.pyplot as plt # Import matplotlib.pyplot
import torch.optim as optim # Import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import ReduceLROnPlateau

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data (using DATA_DIR from config.py)
    segments, labels = load_and_preprocess_data(DATA_DIR)

    print(f"Total segments: {len(segments)}")
    print(f"Normal segments: {sum(labels == 0)}")
    print(f"Abnormal segments: {sum(labels == 1)}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Use BATCH_SIZE from config
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Use BATCH_SIZE from config
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Use BATCH_SIZE from config

    # Initialize model and training components
    model = AttentionECGModel(input_shape=(3600, 1), num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Use LEARNING_RATE from config
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device) # Use NUM_EPOCHS from config

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, device)

    # Visualize attention weights
    visualize_attention_enhanced(model, test_loader, device)