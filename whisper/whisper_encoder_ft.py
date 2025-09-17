import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperModel, WhisperProcessor, WhisperConfig
from tqdm import tqdm
import torchaudio
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F

MODEL_NAME = "openai/whisper-base"
NUM_CLASSES = 11
NUM_UNFROZEN_LAYERS = 4
BATCH_SIZE = 16
LR = 1e-5
NUM_EPOCHS = 5
TARGET_SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

def compute_eer(pairs, numP, numN):
    all_scores = sorted(pairs, key=lambda x: x[0])
    numFA, numFR = numN, 0
    best_eer = 0.0
    best_thresh = 0.0
    memory = []
    for score, is_target in all_scores:
        if is_target:
            numFR += 1
        else:
            numFA -= 1
        
        far = numFA/numN
        frr = numFR/numP
        if far<= frr:
            delta = abs(far + frr) / 2
            prev_delta = abs(memory[0]-memory[1]) if memory else float('inf')
            if delta <= prev_delta:
                best_eer = (far + frr)/2
                best_thresh = score
            else:
                best_eer = (memory[0]+memory[1])/2
                best_thresh = memory[0]

            return best_eer, best_thresh
        memory = [far, frr, score]
    return best_eer, best_thresh
            
def read_txt(file_path):
    recs = []
    with open(file_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            recs.append({'audio_path': path, 'label': label})
        return recs
    
train_recs = read_txt("train_raw_wav_num.txt")
test_recs = read_txt("test_raw_wav_num.txt")

le = LabelEncoder()
all_lbls = [r['label'] for r in train_recs + test_recs]
le.fit(all_lbls)
for r in train_recs: r['label'] = int(le.transform([r['label']])[0])
for r in test_recs: r['label'] = int(le.transform([r['label']])[0])

#dataset and dataloader
class WhisperAudioDataset(Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.recs[idx]['audio_path'])
        if sr!= TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return wav.squeeze(0), self.recs[idx]['label']
    
def collect_fn(batch):
    wvs, lbls = zip(*batch)
    wvs_np = [w.numpy() for w in wvs]
    inp = processor(wvs_np, sampling_rate=TARGET_SR, return_tensors="pt", padding='longest', truncation=True)
    feats = inp.input_features
    T = feats.size(2)
    if T < 3000:
        feats = F.pad(feats, (0, 3000-T))
    else:
        feats = feats[:, :, :3000]
    return feats, torch.tensor(lbls).long()

train_loader = DataLoader(WhisperAudioDataset(train_recs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collect_fn)
test_loader = DataLoader(WhisperAudioDataset(test_recs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collect_fn)

#model defination

class FineTuneWhisperEncoder(nn.Module):
    def __init__(self):
        super().__init__()  # Correctly initialize the parent class
        self.config = WhisperConfig.from_pretrained(MODEL_NAME)  # Assign to self.config
        self.whisper = WhisperModel.from_pretrained(MODEL_NAME)
        for p in self.whisper.parameters(): 
            p.requires_grad = False
        for layer in self.whisper.encoder.layers[-NUM_UNFROZEN_LAYERS:]:
            for p in layer.parameters(): 
                p.requires_grad = True
            
        self.classifier = nn.Sequential(  # Fix typo: nn.squential -> nn.Sequential
            nn.Linear(self.config.d_model, 256), 
            nn.ReLU(), 
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        out = self.whisper.encoder(x).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)
    
model = FineTuneWhisperEncoder().to(device)
optimizer =  optim.AdamW([
    {'params': model.whisper.encoder.layers[-NUM_UNFROZEN_LAYERS:].parameters()},
    {'params': model.classifier.parameters()}
], lr = LR)
criterion = nn.CrossEntropyLoss()

## training

best_bal_acc = 0.0
best_eer = float('inf')
print(model)

for epoch in range(1,NUM_EPOCHS+1):
    model.train()
    train_loss = 0.0
    for feats, labels in tqdm(train_loader, desc='Train'):
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    all_preds, all_lbls = [], []
    pairs, pairs2 = [], []
    with torch.no_grad():
        for feats, labels in tqdm(test_loader, desc='Eval'):
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            val_loss += criterion(logits, labels).item()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(labels.cpu().numpy())

            for i, true in enumerate(labels.cpu().numpy()):
                for lang in range(NUM_CLASSES):
                    sc = probs[i, lang].item()
                    pairs.append((true, lang, sc))
                    pairs2.append((sc, 1 if lang==true else 0))

    val_loss /= len(test_loader)
    bal_acc = balanced_accuracy_score(all_lbls, all_preds)
    eer, thr = compute_eer(pairs2, len(all_lbls), len(all_lbls)*(NUM_CLASSES-1))
    
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        torch.save(model.state_dict(), "model_11_lang_whisperft_18_06.pth")

