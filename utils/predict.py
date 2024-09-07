import torch
from data.preprocessing import bandpass_filter
from sklearn.preprocessing import StandardScaler
import numpy as np

def predict(model, ecg_signal, device, class_names=None):
    """
    Makes a prediction for a single ECG signal using the trained model.
    
    Args:
        model: The trained PyTorch model.
        ecg_signal: The raw ECG signal (1D NumPy array).
        device: The device (CPU or GPU) to use for prediction.
        class_names: (Optional) A list of class names 
                     (e.g., ["Normal", "Abnormal"]).

    Returns:
        The predicted class label (either an index or a class name if 
        class_names are provided).
    """
    # Preprocess the ECG signal (same preprocessing as during training)
    ecg_signal = bandpass_filter(ecg_signal, lowcut=0.5, highcut=50, fs=360, order=5)
    segment_length = 3600  # Adjust if needed
    if len(ecg_signal) < segment_length:
        # Handle cases where the signal is shorter than the expected segment length
        ecg_signal = np.pad(ecg_signal, (0, segment_length - len(ecg_signal)), 'constant')
    segment = ecg_signal[:segment_length]  # Take the first segment
    scaler = StandardScaler()
    segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(1, 1, segment_length)

    # Convert the segment to a PyTorch tensor and move it to the device
    input_tensor = torch.FloatTensor(segment).to(device)

    # Make the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output, _ = model(input_tensor)
        predicted_class_index = output.argmax(1).item()

    # Return the prediction (index or class name)
    if class_names:
        return class_names[predicted_class_index]
    else:
        return predicted_class_index