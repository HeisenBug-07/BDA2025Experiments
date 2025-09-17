import numpy as np
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils


def collate_fn_atten(batch):
    """
    Collate function for attention-based models.

    Sorts a batch (list of tuples) by sequence length (descending), pads sequences,
    and converts labels to a tensor.

    Args:
        batch (list): List of tuples (feature, label, seq_len).

    Returns:
        Tuple:
            - data (Tensor): Padded feature sequences of shape [batch, max_seq_len, feature_dim].
            - labels (Tensor): LongTensor of labels.
            - seq_length (tuple): Original sequence lengths for each sample.
    """
    # Sort by descending sequence length
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data_padded = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data_padded, labels, seq_length


class RawFeatures(data.Dataset):
    """
    Dataset for raw features stored in files.

    The provided text file should have one sample per line, with the first token
    as the file path (to a .npy file) and the second token as the label.
    """

    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [line.split()[0] for line in lines]
            self.label_list = [line.split()[1] for line in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        # Load the numpy array and transpose it so that time steps are in dimension 0
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True).T)
        seq_len = int(feature.size(0))  # Save sequence length as an integer
        label = int(self.label_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)


def get_atten_mask(seq_lens, batch_size):
    """
    Create a square attention mask for a batch based on sequence lengths.
    Positions outside the actual sequence lengths are masked.

    Args:
        seq_lens (list or tuple): Sequence lengths for each sample (already sorted in descending order).
        batch_size (int): The batch size.

    Returns:
        Tensor: A boolean attention mask of shape [batch_size, max_len, max_len].
    """
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0  # Valid positions
    return atten_mask.bool()


class PairedDataset(data.Dataset):
    """
    Dataset for paired utterances.

    Expects a text file where each line contains three tokens: the path to the first utterance,
    the path to the second utterance, and the corresponding label (as an integer).
    """

    def __init__(self, txt_path):
        self.pairs = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 3:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                utt1_path, utt2_path, label = parts[0], parts[1], int(parts[2])
                self.pairs.append((utt1_path, utt2_path, label))

    def __getitem__(self, index):
        utt1_path, utt2_path, label = self.pairs[index]
        utt1 = torch.from_numpy(np.load(utt1_path, allow_pickle=True).T)
        utt2 = torch.from_numpy(np.load(utt2_path, allow_pickle=True).T)
        return utt1, utt2, label

    def __len__(self):
        return len(self.pairs)


def collate_fn_paired(batch):
    """
    Collate function for paired datasets.

    Pads both utterance sequences in the batch and returns the padded sequences,
    the labels, and their lengths.

    Args:
        batch (list): List of tuples (utt1, utt2, label).

    Returns:
        Tuple:
            - (utt1_batch, utt2_batch): Tuple of padded utterance tensors.
            - labels (Tensor): FloatTensor of labels.
            - seq_lengths (list): List of sequence lengths (based on first utterance).
    """
    utt1_batch, utt2_batch, labels = [], [], []
    seq_lengths = []  # Store original sequence lengths for utterance 1
    for utt1, utt2, label in batch:
        utt1_batch.append(utt1)
        utt2_batch.append(utt2)
        labels.append(label)
        seq_lengths.append(utt1.size(0))
    utt1_batch = rnn_utils.pad_sequence(utt1_batch, batch_first=True, padding_value=0)
    utt2_batch = rnn_utils.pad_sequence(utt2_batch, batch_first=True, padding_value=0)
    labels = torch.FloatTensor(labels)
    return (utt1_batch, utt2_batch), labels, seq_lengths
