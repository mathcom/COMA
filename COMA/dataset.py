import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BasicSmilesDataset(Dataset):
    def __init__(self, device=None):
        super(BasicSmilesDataset, self).__init__()
        ## Basic vocabulary
        self.sos_char = "<"
        self.eos_char = ">"
        self.pad_char = "_"
        self.unk_char = "?"
        self.char2idx = {self.pad_char:0, self.sos_char:1, self.eos_char:2, self.unk_char:3}
        self.idx2char = [self.pad_char, self.sos_char, self.eos_char, self.unk_char]
        self.vocab_size = len(self.char2idx)
        self.device = torch.device('cpu') if device is None else device
        
    @property
    def sos_idx(self):
        return self.char2idx[self.sos_char]
    
    @property
    def eos_idx(self):
        return self.char2idx[self.eos_char]

    @property
    def pad_idx(self):
        return self.char2idx[self.pad_char]
        
    @property
    def unk_idx(self):
        return self.char2idx[self.unk_char]
        
    def encode(self, smiles, max_seqlen):
        """Transform SMILES-strings into one-hot tensors.
        
        Parameters
        ----------
        smiles : a list of strings (batch_size, )
        max_seqlen : a maximum length of strings
        
        Returns
        -------
        res : tensor of shape (batch_size, max_seqlen)
        """
        batch_size = len(smiles)
        
        res = np.zeros((batch_size, max_seqlen), dtype=np.int64)
        for i, smi in enumerate(smiles):
            for j, c in enumerate(smi):
                res[i][j] = self.char2idx.get(c, self.unk_idx)
                
        res = torch.Tensor(res).long().to(self.device)
        return res
        
    def decode(self, encoded, trim=True): # encoded : list of indices
        """Transform one-hot vectors into SMILES-strings.
        
        Parameters
        ----------
        encoded : array-like of shape (batch_size, max_seqlen)
        
        Returns
        -------
        res : a list of strings (batch_size, )
        """
        batch_size = len(encoded)
        
        res = []
        for i in range(batch_size):
            smi = self._decode(encoded[i])
            if trim: smi = smi.replace(self.sos_char, "").replace(self.eos_char, "")
            res.append(smi)
        return res
        
    def _decode(self, indices):
        tokens = []
        for i in indices:
            if (i == self.pad_idx): break
            elif (i == self.unk_idx): break
            elif (i == self.eos_idx): tokens.append(self.idx2char[i]); break
            else: tokens.append(self.idx2char[i])
        smi = "".join(tokens)
        return smi
        
    def _load_vocab(self, filepath_char2idx):
        df_char2idx = pd.read_csv(filepath_char2idx, index_col=0)
        idx2char = df_char2idx.iloc[:,0].values.tolist()
        char2idx = {c:i for i,c in enumerate(idx2char)}
        return char2idx, idx2char


class TrainingSmilesDataset(BasicSmilesDataset):
    def __init__(self, filepath, filepath_char2idx=None, device=None):
        """PyTorch dataset class for MTMR training.
        
        Parameters
        ----------
        filepath : SMILES triplet dataset having three columns (src, tar, neg), but no header
        filepath_char2idx : (optional) vocabulary for SMILES having two columns (char, idx), but no header
        device : (optional) Location for PyTorch neural network. Default is CPU
        """
        super(TrainingSmilesDataset, self).__init__(device)
        ## Read a file
        self.df = pd.read_csv(filepath, sep=" ", header=None).rename(columns={0:"smiles_src", 1:"smiles_tar", 2:"smiles_neg"})
        self.num_smiles = len(self.df)
        ## Construct vocabulary
        if filepath_char2idx is None:
            self.char2idx, self.idx2char = self._update_vocab()
        else:
            self.char2idx, self.idx2char = self._load_vocab(filepath_char2idx)
        self.vocab_size = len(self.char2idx)

    def __len__(self):
        return self.num_smiles
        
    def __getitem__(self, idx):
        batch_smiles_A = self.sos_char + self.df["smiles_src"][idx] + self.eos_char
        batch_smiles_B = self.sos_char + self.df["smiles_tar"][idx] + self.eos_char
        batch_smiles_C = self.sos_char + self.df["smiles_neg"][idx] + self.eos_char
        batch_length_A = len(batch_smiles_A)
        batch_length_B = len(batch_smiles_B)
        batch_length_C = len(batch_smiles_C)
        return {"smiles_s": batch_smiles_A,
                "length_s": batch_length_A,
                "smiles_t": batch_smiles_B,
                "length_t": batch_length_B,
                "smiles_n": batch_smiles_C,
                "length_n": batch_length_C}

    def _update_vocab(self):
        ## init
        char2idx = self.char2idx.copy() # initialized in the super-class
        idx2char = self.idx2char.copy() # initialized in the super-class
        ## visit all strings
        t = len(char2idx) # start index
        for smiles in [self.df["smiles_src"], self.df["smiles_tar"], self.df["smiles_neg"]]:
            for smi in smiles:
                for char in smi:
                    if char not in char2idx:
                        char2idx[char] = t
                        idx2char.append(char)
                        t += 1
        return char2idx, idx2char
        
    def save_char2idx(self, filepath):
        with open(filepath, "w") as fout:
            fout.write(",char\n")
            for i, c in enumerate(self.idx2char):
                fout.write(f"{i},{c}\n")
        
    def get_targets(self):
        return self.df.loc[:,"smiles_tar"].drop_duplicates().values.tolist()
        
    def head(self, *args, **kwargs):
        return self.df.head(*args, **kwargs)
        
    @property
    def shape(self):
        return self.df.shape


class ValidationSmilesDataset(BasicSmilesDataset):
    def __init__(self, filepath, filepath_char2idx, device=None):
        """PyTorch dataset class for MTMR training.
        
        Parameters
        ----------
        filepath : SMILES triplet dataset having three columns (src, tar, neg), but no header
        filepath_char2idx : vocabulary constructed by SmilesTrainingDataset
        device : (optional) Location for PyTorch neural network. Default is CPU
        """
        super(ValidationSmilesDataset, self).__init__(device)
        ## Read a file
        self.df = pd.read_csv(filepath, header=None).rename(columns={0:"smiles_src"})
        self.num_smiles = len(self.df)
        ## Construct vocabulary
        self.char2idx, self.idx2char = self._load_vocab(filepath_char2idx)
        self.vocab_size = len(self.char2idx)
        
    def __len__(self):
        return self.num_smiles
        
    def __getitem__(self, idx):
        batch_smiles = self.sos_char + self.df["smiles_src"][idx] + self.eos_char
        batch_length = len(batch_smiles)
        return {"smiles_s": batch_smiles,
                "length_s": batch_length}

    def head(self, *args, **kwargs):
        return self.df.head(*args, **kwargs)
        
    @property
    def shape(self):
        return self.df.shape