import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

# -------------------------
# Dataset Class (Unchanged)
# -------------------------
class FeatureDataset(Dataset):
    def __init__(self, txt_file, root_dir=""):
        self.root_dir = root_dir
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    feat_path = parts[0]
                    label = int(parts[1])
                    self.samples.append((feat_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feat = np.load(os.path.join(self.root_dir, feat_path))
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -------------------------
# Collate Function (Unchanged)
# -------------------------
def collate_fn(batch, fixed_time_dim=1500):
    features, labels = zip(*batch)
    padded_features = torch.zeros(len(features), 392, fixed_time_dim)
    adjusted_lengths = []
    for i, f in enumerate(features):
        time_length = f.shape[1]
        if time_length >= fixed_time_dim:
            padded_features[i] = f[:, :fixed_time_dim]
            adjusted_lengths.append(fixed_time_dim)
        else:
            padded_features[i, :, :time_length] = f
            adjusted_lengths.append(time_length)
    return padded_features, torch.stack(labels), torch.tensor(adjusted_lengths)

# -------------------------
# Modified Model with Debugging
# -------------------------
class CustomECAPA(nn.Module):
    def __init__(self, classifier, num_classes, freeze_encoder_except_last_n=4):
        super().__init__()
        # Extract pretrained components
        self.encoder = classifier.mods.embedding_model
        self.mean_var_norm = classifier.mods.mean_var_norm
        
        # Conv1d projection (392 -> 60 channels)
        self.projection = nn.Conv1d(392, 60, kernel_size=1)
        
        # Final classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Freeze the encoder except for the last `n` layers
        self.freeze_encoder_except_last_n(freeze_encoder_except_last_n)

    def freeze_encoder_except_last_n(self, n):
        # Freeze all layers except for the last `n` layers
        encoder_layers = list(self.encoder.children())
        for i in range(len(encoder_layers) - n):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

    def forward(self, x, lengths=None):
        # Input shape: (batch, 392, time)
        
        # 1. Channel projection
        x = self.projection(x)
        
        # 2. Prepare for ECAPA (time last)
        x = x.transpose(1, 2)  # (B, T, 60)
        
        # 3. Pass through ECAPA
        x = self.encoder(x)  # (B, 1, 256)
        
        # 4. Remove the extra dimension
        x = x.squeeze(1)  # (B, 256)
        
        # 5. Normalization
        x = self.mean_var_norm(x, lengths=lengths)
        
        # 6. Final classification
        logits = self.classifier(x)  # (B, num_classes)
        return logits

# -------------------------
# Modified Training Loop
# -------------------------
def main():
    # Initialize datasets
    train_dataset = FeatureDataset("train_phone_md.txt")
    valid_dataset = FeatureDataset("test_phone_md.txt")
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, fixed_time_dim=1500)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, fixed_time_dim=1500)
    )
    
    # Load pretrained model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="pretrained_models"
    )
    
    # Initialize custom model
    model = CustomECAPA(classifier, num_classes=12, freeze_encoder_except_last_n=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.projection.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels, lengths = batch
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, lengths=lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                inputs, labels, lengths = batch
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                
                outputs = model(inputs, lengths=lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = correct / total
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"ecapa_phonemd_{val_acc:.2f}_{epoch+1}.pth")

if __name__ == "__main__":
    main()
