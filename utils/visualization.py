import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import resample
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def visualize_attention(model, dataloader, device, num_samples=3, class_names=None):
    """
    Visualizes the attention weights learned by the model for a few samples 
    from the dataloader, providing a clearer interpretation of the model's 
    focus on different parts of the ECG signal.

    Args:
        model: The trained PyTorch model with attention mechanism.
        dataloader: DataLoader for the dataset (e.g., test set).
        device: The device (CPU or GPU) to use.
        num_samples: The number of samples to visualize.
        class_names: (Optional) A list of class names to use in the plot titles 
                     (e.g., ["Normal", "Abnormal"]). If not provided, class 
                     indices will be used.
    """
    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_samples, 1, figsize=(20, 4 * num_samples))

    with torch.no_grad():  
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_samples:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attention_weights = model(inputs)

            # Get the first sample's ECG signal and attention weights 
            ecg_signal = inputs[0, 0].cpu().numpy()
            attention = attention_weights[0, 0].cpu().numpy() 

            # Upsample attention weights to match the ECG signal length
            attention_upsampled = resample(attention, len(ecg_signal))

            # Get the predicted class and true label (with optional class names)
            predicted_class = outputs.argmax(1)[0].item()
            true_label = labels[0].item()
            if class_names:
                predicted_class = class_names[predicted_class]
                true_label = class_names[true_label]

            # Plot the ECG signal with attention highlighted
            axs[i].plot(ecg_signal, color='blue', linewidth=1.5, label='ECG Signal')
            axs[i].set_title(f'Sample {i+1}: True Label = {true_label}, Predicted = {predicted_class}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Amplitude')

            # Overlay the attention as a shaded region 
            axs[i].fill_between(range(len(ecg_signal)), ecg_signal, 
                                where=(attention_upsampled > np.percentile(attention_upsampled, 95)),
                                color='red', alpha=0.3, label='High Attention')

            axs[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()