import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import WhisperModel, WhisperConfig
import librosa

# Custom EER & Cavg Functions
def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
    cavgs = [0.0] * (bins + 1)
    precision = (max_score - min_score) / bins
    for section in range(bins + 1):
        threshold = min_score + section * precision
        target_cavg = [0.0] * lang_num
        for lang in range(lang_num):
            p_miss, LTa, LTm = 0.0, 0.0, 0.0
            p_fa = [0.0] * lang_num
            LNa = [0.0] * lang_num
            LNf = [0.0] * lang_num
            for line in pairs:
                if line[0] == lang:
                    if line[1] == lang:
                        LTa += 1
                        if line[2] < threshold:
                            LTm += 1
                    else:
                        LNa[line[1]] += 1
                        if line[2] >= threshold:
                            LNf[line[1]] += 1
            if LTa != 0.0:
                p_miss = LTm / LTa
            for i in range(lang_num):
                if LNa[i] != 0.0:
                    p_fa[i] = LNf[i] / LNa[i]
            p_nontarget = (1 - p_target) / (lang_num - 1)
            target_cavg[lang] = p_target * p_miss + p_nontarget * sum(p_fa)
        cavgs[section] = sum(target_cavg) / lang_num
    return cavgs, min(cavgs)

def compute_eer(pairs_2, numP, numN):
    allScores = sorted(pairs_2, reverse=False)
    numFA, numFR = numN, 0
    eer, threshold = 0.0, 0.0
    memory = []
    for tuple in allScores:
        if tuple[1] == 1:
            numFR += 1
        else:
            numFA -= 1
        far = numFA * 1.0 / numN
        frr = numFR * 1.0 / numP
        if far <= frr:
            lnow = abs(far - frr)
            lmemory = abs(memory[0] - memory[1])
            if lnow <= lmemory:
                eer = (far + frr) / 2
                threshold = tuple[0]
            else:
                eer = (memory[0] + memory[1]) / 2
                threshold = memory[2]
            return eer, threshold
        else:
            memory = [far, frr, tuple[0]]

# Configuration
MODEL_NAME = "openai/whisper-base"
NUM_CLASSES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ekstepmandi_whisper_raw_epoch5.pt"
TEST_TXT = "ekstepmandi_12_test_raw.txt"
MODEL_ID = "whisper_inference"

# Load and preprocess data
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            audio_path, label = line.strip().split()
            data.append({"audio_path": audio_path, "label": label})
    return data

test_data_raw = read_txt_file(TEST_TXT)
label_encoder = LabelEncoder()
all_labels = [item["label"] for item in test_data_raw]
label_encoder.fit(all_labels)
for item in test_data_raw:
    item["label"] = label_encoder.transform([item["label"]])[0]

class WhisperRawAudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.data[idx]["audio_path"], sr=16000)
        wav = torch.tensor(wav, dtype=torch.float32)
        return wav, torch.tensor(self.data[idx]["label"], dtype=torch.long)

def collate_fn(batch):
    audios, labels = zip(*batch)
    max_len = max([x.shape[0] for x in audios])
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, :a.shape[0]] = a
    return padded, torch.tensor(labels, dtype=torch.long)

test_dataset = WhisperRawAudioDataset(test_data_raw)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Model definition
class FineTunedWhisperLID(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = WhisperConfig.from_pretrained(MODEL_NAME)
        self.whisper = WhisperModel.from_pretrained(MODEL_NAME)

        for param in self.whisper.parameters():
            param.requires_grad = False
        for layer in self.whisper.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        outputs = self.whisper.encoder(x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# Evaluation

def evaluate(model, data_loader, data_raw, num_classes, model_name):
    model.eval()
    all_outputs = []
    all_predictions = []
    ground_truth = []

    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
            features = features.to(DEVICE)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            ground_truth.extend(labels.numpy())

    scores = np.vstack(all_outputs)
    ground_truth = np.array(ground_truth)
    predictions_array = np.array(all_predictions)

    bal_acc = balanced_accuracy_score(ground_truth, predictions_array)
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Format pairs for EER/Cavg
    pairs = []
    num_pos, num_neg = 0, 0
    for i, entry in enumerate(data_raw):
        true_label = entry["label"]
        for j in range(num_classes):
            score = scores[i][j]
            if j == true_label:
                pairs.append((true_label, j, score))
                num_pos += 1
            else:
                pairs.append((true_label, j, score))
                num_neg += 1

    pairs_for_eer = [(s[2], 1 if s[0] == s[1] else 0) for s in pairs]
    eer, threshold = compute_eer(pairs_for_eer, num_pos, num_neg)
    print(f"Custom EER: {eer:.4f} @ threshold={threshold:.4f}")

    scores_only = [s[2] for s in pairs]
    cavgs, min_cavg = get_cavg(pairs, num_classes, min(scores_only), max(scores_only))
    print(f"Custom Cavg: {min_cavg:.4f}")

    return predictions_array, scores, eer, min_cavg

# Run
if __name__ == "__main__":
    model = FineTunedWhisperLID().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded. Starting evaluation...\n")

    predictions, scores, overall_eer, overall_cavg = evaluate(
        model, test_loader, test_data_raw, NUM_CLASSES, MODEL_ID
    )

    print("\nEvaluation complete.")
    print(f"Final EER: {overall_eer:.4f}")
    print(f"Final Cavg: {overall_cavg:.4f}")