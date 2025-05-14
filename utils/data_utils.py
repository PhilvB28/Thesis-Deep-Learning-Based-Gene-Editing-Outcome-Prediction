import numpy as np
import torch
from torch.utils.data import Dataset

from utils.encoders import dna_to_onehot, dna_to_onehot_extra, dna_to_kmer_onehot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class CRISPRDataset_557(Dataset):
    def __init__(self, df, sequence_length=60, device ="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sequences = df['target_seq'].tolist()

        labels = df.iloc[:, -557:].values.astype(np.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32,device=self.device)

        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # One-hot encode
        #onehot = dna_to_onehot_extra([sequence], self.sequence_length)
        onehot = dna_to_onehot([sequence], self.sequence_length)
        onehot = onehot.to(self.device)

        return onehot, label


class CRISPRDataset(Dataset):
    def __init__(self, df, sequence_length=60, device ="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length

        self.sequences = df['sequences'].tolist()

        # Label 1: Deletion label
        label_del = df['deletion frequency'].tolist()
        # Label 2: 1bp Deletion labels
        label_1bpdel = df['1 bp deletion frequency'].tolist()
        # Label 3: 1bp Insertion labels
        label_1bpins = df['1 bp insertion frequency'].tolist()
        # Label 4: 1bp frameshift frequency
        label_1bpfreq = df['1 bp frameshift frequency'].tolist()
        # Label 5: 2bp frameshift frequency
        label_2bpfreq = df['2 bp frameshift frequency'].tolist()
        # Label 6: total frameshift frequency
        label_totalfreq = df['frameshift frequency'].tolist()


        labels = np.column_stack([label_del, label_1bpdel, label_1bpins, label_1bpfreq, label_2bpfreq, label_totalfreq])

        # Convert the labels array to a torch tensor
        self.labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # One-hot encode
        onehot = dna_to_onehot([sequence], self.sequence_length)
        #onehot = dna_to_kmer_onehot([sequence], k=2, sequence_length = self.sequence_length)
        onehot = onehot.to(self.device)

        #print("Debugging Info")
        #print("ONEHOT:", onehot)
        #print("Label:", label)

        return onehot, label
