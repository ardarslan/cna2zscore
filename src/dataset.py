import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any], input_file_name: str, output_file_name: str):
        super().__init__()
        self.cfg = cfg

        split_ratios = cfg["split_ratios"]

        if "train" not in split_ratios.keys() or "val" not in split_ratios.keys() or "test" not in split_ratios.keys():
            raise Exception("The dictionary 'split_ratios' should have the following keys: 'train', 'val' and 'test'.")

        if split_ratios["train"] + split_ratios["val"] + split_ratios["test"] != 1.0:
            raise Exception("The values of dictionary 'split_ratios' should sum up to 1.0.")

        processed_data_dir = cfg["processed_data_dir"]
        seed = cfg["seed"]

        input_df = pd.read_csv(os.path.join(processed_data_dir, input_file_name), sep="\t", index_col=0)
        output_df = pd.read_csv(os.path.join(processed_data_dir, output_file_name), sep="\t", index_col=0)
        merged_df = pd.merge(input_df, output_df, how="inner", left_index=True, right_index=True)
        merged_df = shuffle(merged_df, random_state=seed)
        self.X = merged_df[merged_df.columns[:input_df.shape[1]]].values
        self.y = merged_df[merged_df.columns[input_df.shape[1]:]].values

        self.input_dimension = self.X.shape[1]
        self.output_dimension = self.y.shape[1]

        self.device = cfg["device"]
        self.len_dataset = self.X.shape[0]

        all_indices = shuffle(range(self.len_dataset), random_state=seed)
        train_size = int(self.len_dataset * split_ratios["train"])
        val_size = int(self.len_dataset * split_ratios["val"])

        self.train_idx = all_indices[:train_size]
        self.val_idx = all_indices[train_size:train_size+val_size]
        self.test_idx = all_indices[train_size+val_size:]

        X_train = self.X[self.train_idx, :]
        y_train = self.y[self.train_idx, :]

        self.X_train_mean = torch.as_tensor(np.mean(X_train, axis=0), device=self.device)
        self.X_train_std = torch.as_tensor(np.std(X_train, axis=0), device=self.device)

        self.y_train_mean = torch.as_tensor(np.mean(y_train, axis=0), device=self.device)
        self.y_train_std = torch.as_tensor(np.std(y_train, axis=0), device=self.device)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
                "X": torch.as_tensor(self.X[idx, :], device=self.device, dtype=torch.float32),
                "y": torch.as_tensor(self.y[idx, :], device=self.device, dtype=torch.float32)
               }

    def __len__(self) -> int:
        return self.len_dataset


class RPPA2RNADataset(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg=cfg, input_file_name="rppa_df.tsv", output_file_name="rna_df.tsv")


class RNA2RNADataset(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg=cfg, input_file_name="rna_df.tsv", output_file_name="rna_df.tsv")


class AverageRNAClinicalSubtype2RNADataset(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg=cfg, input_file_name="avg_rna_clinical_subtype_df.tsv", output_file_name="rna_df.tsv")
