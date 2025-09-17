import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils

def collate_fn_atten(batch):
    """Collate function for attention-based models.
    
    Sorts the batch by descending sequence length, pads the feature tensors,
    and returns the padded features, language labels, and original sequence lengths.
    """
    # Sort by sequence length (longest first)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # Unpack elements
    features, labels, seq_lens = zip(*batch)
    
    # Pad sequences: output shape is [batch_size, max_seq_len, 392]
    padded_features = rnn_utils.pad_sequence(
        features, 
        batch_first=True, 
        padding_value=0
    )
    
    return (
        padded_features,         # Padded features tensor
        torch.LongTensor(labels),# Language labels
        list(seq_lens)           # Original sequence lengths
    )

class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        """Initialize the dataset with the path to the training data.
        
        The provided text file should have one sample per line, with the feature file
        path and corresponding label separated by whitespace.
        """
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [line.split()[0] for line in lines]
            self.label_list = [line.split()[1] for line in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        # Load feature from .npy file and transpose to [T, 392]
        feature = np.load(feature_path).T
        return (
            torch.FloatTensor(feature),  # Feature tensor with shape [T, 392]
            int(self.label_list[index]), # Language label as integer
            feature.shape[0]             # Sequence length (T)
        )

    def __len__(self):
        return len(self.feature_list)

def get_atten_mask(seq_lens, batch_size):
    """Create an attention mask for a transformer encoder.
    
    For each sample in the batch, valid positions (i.e. within the sequence length)
    are marked as False (meaning they are *not* masked), and positions outside are True.
    This is useful for transformer implementations where True indicates positions to mask.
    """
    max_len = max(seq_lens)
    mask = torch.ones(batch_size, max_len, max_len, dtype=torch.bool)
    
    for i, length in enumerate(seq_lens):
        mask[i, :length, :length] = False  # Mark valid positions as not masked
        
    return mask

def get_atten_mask_student(seq_lens, batch_size, mask_type='fix', win_len=15):
    """Create a masked attention map for the student model.
    
    Depending on the mask_type, a fixed window or a random window of positions 
    is unmasked (set to False) for each sample.
    """
    max_len = max(seq_lens)
    mask = torch.ones(batch_size, max_len, max_len, dtype=torch.bool)
    
    for i in range(batch_size):
        seq_len = seq_lens[i]
        if mask_type == 'fix':
            if seq_len > win_len:
                mask[i, :win_len, :win_len] = False
            else:
                mask[i, :seq_len, :seq_len] = False
        elif mask_type == 'random':
            if seq_len > win_len:
                start = random.randint(0, seq_len - win_len)
                mask[i, start:start+win_len, start:start+win_len] = False
            else:
                mask[i, :seq_len, :seq_len] = False
                
    return mask
