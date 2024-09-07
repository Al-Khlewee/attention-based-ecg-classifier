# Attention-Based ECG Classification

This project uses deep learning with an attention mechanism to classify ECG signals and detect potential cardiac abnormalities.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Deployment (API)](#deployment-api)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Electrocardiogram (ECG) signals provide valuable insights into the electrical activity of the heart and are widely used for diagnosing various cardiac conditions. This project aims to develop an accurate and interpretable ECG classification system using deep learning with an attention mechanism. 

The project utilizes a convolutional neural network (CNN) combined with a multi-head attention mechanism to effectively capture temporal dependencies and salient features within ECG signals. The attention mechanism allows the model to focus on specific segments of the ECG that are most relevant for classification, improving both performance and interpretability.

The goal of this project is to achieve high accuracy in classifying different types of arrhythmias, paving the way for more efficient and reliable automated ECG analysis tools.

## Dataset

The project utilizes the MIT-BIH Arrhythmia Database, a publicly available dataset widely used for ECG classification research. It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. 

The dataset includes annotations for various types of heartbeats, including normal sinus rhythm, atrial fibrillation, ventricular premature beats, and more. 

**Preprocessing:**

The following preprocessing steps were applied to the ECG signals:

- **Bandpass Filtering:** A bandpass filter with a cutoff frequency of 0.5-50 Hz was applied to remove baseline wander and high-frequency noise.
- **Segmentation:** The ECG signals were segmented into 10-second (3600 samples) segments with a 50% overlap.
- **Standardization:** Each segment was standardized to have zero mean and unit variance.
- **Labeling:** Each segment was labeled based on the majority annotation within the segment. Segments with a majority of abnormal beats were labeled as "Abnormal," while segments with a majority of normal beats were labeled as "Normal."

## Model Architecture

The project employs a CNN-based model with a multi-head attention mechanism. The model consists of the following layers:

1. **Convolutional Layers:** Two convolutional layers with 32 and 64 filters, respectively, followed by ReLU activation and max pooling. These layers extract local features from the ECG signal.
2. **Multi-Head Attention:** A multi-head attention layer with 8 heads is applied to capture long-range dependencies and focus on important parts of the ECG signal.
3. **Flatten Layer:** The output of the attention layer is flattened.
4. **Fully Connected Layers:** Two fully connected layers with 128 and 2 neurons, respectively. The final layer uses a softmax activation function to output the probability of each class (Normal or Abnormal).
5. **Dropout:** Dropout is used after the first fully connected layer to prevent overfitting.

**[Optional: Include a diagram of your model architecture.]**

## Features

- **Attention Mechanism:** The multi-head attention mechanism provides improved interpretability by highlighting the parts of the ECG signal that are most relevant for classification.
- **Efficient Implementation:** The model is implemented using PyTorch, a powerful deep learning framework, allowing for efficient training and inference.
- **Data Visualization:** The project includes code for visualizing the attention weights, providing insights into the model's decision-making process.

## Usage

### Dependencies

- Python 3.x
- PyTorch 
- NumPy
- SciPy
- Scikit-learn
- wfdb
- FastAPI
- uvicorn
- Matplotlib
- Seaborn

### Installation

1. Clone the repository: `git clone https://github.com/your-username/your-repository-name.git`
2. Create a virtual environment (recommended): `python -m venv .venv`
3. Activate the virtual environment: `. .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
4. Install the requirements: `pip install -r requirements.txt`

### Training

1. Download the MIT-BIH Arrhythmia Database and place it in the `./mit-bih-arrhythmia-database-1.0.0/` directory.
2. Run the training script: `python main.py`
   - Configuration options (e.g., batch size, number of epochs) can be modified in the `config.py` file.

### Evaluation

1. Load the best saved model (located in `best_model.pth`).
2. Run the evaluation part of the `main.py` script.
   - The script will evaluate the model on the test set and display the results, including accuracy, precision, recall, F1-score, and a confusion matrix.

### Deployment (API)

1. Run the API: `uvicorn api:app --reload`
2. Send a POST request to the `/predict` endpoint with an ECG signal in a `.dat` file as the payload.
   - The API will return the predicted class (Normal or Abnormal) and the processed ECG signal.



## Future Work

- **Experiment with Different Architectures:** Explore other deep learning architectures, such as recurrent neural networks (RNNs) and transformers, to potentially improve performance.
- **Advanced Attention Mechanisms:** Investigate more sophisticated attention mechanisms, such as self-attention and hierarchical attention, to enhance feature extraction and interpretability.
- **Data Augmentation:** Apply data augmentation techniques to increase the size and diversity of the training data, which could lead to improved generalization.
- **Deployment Optimization:** Develop a more robust and scalable deployment solution, such as integrating the model with a cloud platform or mobile application.

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
content_copy
Use code with caution.
Markdown

Remember to:

Replace placeholders like your-username and your-repository-name with your actual information.

Add specific details about your project's results, including metrics, plots, tables, or figures.

Consider adding a diagram of your model architecture for better visualization.

Tailor the content to accurately reflect your project's unique features and functionalities.
