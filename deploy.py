import torch
import wfdb
from models.attention_ecg_model import AttentionECGModel
from utils.predict import predict 
from config import * # Import your configuration parameters

if __name__ == "__main__":
    # 1. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionECGModel(input_shape=(3600, 1), num_classes=2).to(device)
    model.load_state_dict(torch.load("best_model.pth"))  # Load your saved model weights

    # 2. Load the ECG signal (replace with your signal loading logic)
    record = wfdb.rdrecord("./mit-bih-arrhythmia-database-1.0.0/102") # Replace with your ECG record
    ecg_signal = record.p_signal[:, 0]

    # 3. Make the prediction
    predicted_class = predict(model, ecg_signal, device, class_names=["Normal", "Abnormal"])

    # 4. Display or use the prediction
    print(f"Predicted class: {predicted_class}")