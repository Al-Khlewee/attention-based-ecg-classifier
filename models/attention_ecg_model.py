import torch
import torch.nn as nn

class AttentionECGModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(AttentionECGModel, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_shape[0] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        x, attention_weights = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # (batch, features, seq_len)
        
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, attention_weights 