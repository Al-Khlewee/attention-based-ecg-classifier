import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def assign_label(annotations):
    normal_beats = ['N', '.']
    abnormal_beats = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    
    normal_count = sum(1 for ann in annotations if ann in normal_beats)
    abnormal_count = sum(1 for ann in annotations if ann in abnormal_beats)
    
    return 1 if abnormal_count > normal_count else 0

def load_and_preprocess_data(data_directory):
    record_files = [f[:-4] for f in os.listdir(data_directory) if f.endswith('.dat')]
    segments = []
    labels = []

    for record_name in record_files:
        record_path = os.path.join(data_directory, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        ecg_signal = record.p_signal[:, 0]
        ecg_signal = bandpass_filter(ecg_signal, lowcut=0.5, highcut=50, fs=360, order=5)
        
        segment_length = 3600  # 10 seconds at 360 Hz
        step = segment_length // 2  # 50% overlap

        for i in range(0, len(ecg_signal) - segment_length + 1, step):
            segment = ecg_signal[i:i + segment_length]
            segment_annotations = [
                annotation.symbol[j] 
                for j, sample in enumerate(annotation.sample) 
                if i <= sample < i + segment_length
            ]
            
            label = assign_label(segment_annotations)
            
            segments.append(segment)
            labels.append(label)

    segments = np.array(segments)
    labels = np.array(labels)
    
    scaler = StandardScaler()
    segments = scaler.fit_transform(segments.reshape(-1, segment_length)).reshape(-1, 1, segment_length)

    return segments, labels