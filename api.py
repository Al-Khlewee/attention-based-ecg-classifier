from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
from models.attention_ecg_model import AttentionECGModel
from utils.predict import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - adjust this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionECGModel(input_shape=(3600, 1), num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth"))
class_names = ["Normal", "Abnormal"]

@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded ECG file and return the ECG signal.
    """
    try:
        # 1. Read the uploaded ECG signal data (.dat file)
        contents = await file.read()
        with open("temp.dat", "wb") as f:
            f.write(contents)

        # Read the binary data using NumPy
        ecg_signal = np.fromfile("temp.dat", dtype=np.int16)

        # 2. Make a prediction and preprocess the ECG signal
        predicted_class = predict(model, ecg_signal, device, class_names=class_names)

        # 3. Return the prediction and the processed ECG signal
        return {
            "predicted_class": predicted_class,
            "ecg_signal": ecg_signal.tolist()  # Send back the ECG signal for visualization
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
