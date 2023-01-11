import os
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        input_data_types: List[str],
        output_data_type: str,
        logger: logging.Logger
    ):
        super().__init__()
        self.cfg = cfg
        self.input_data_types = input_data_types
        self.output_data_type = output_data_type
        self.logger = logger

        self.split_ratios = cfg["split_ratios"]

        if "train" not in self.split_ratios.keys() or "val" not in self.split_ratios.keys() or "test" not in self.split_ratios.keys():
            raise Exception("The dictionary 'split_ratios' should have the following keys: 'train', 'val' and 'test'.")

        if self.split_ratios["train"] + self.split_ratios["val"] + self.split_ratios["test"] != 1.0:
            raise Exception("The values of dictionary 'split_ratios' should sum up to 1.0.")

        self.processed_data_dir = cfg["processed_data_dir"]
        self.seed = cfg["seed"]
        self.device = cfg["device"]
        self.logger = logger

        self._process_dataset()

    def _process_dataset(self) -> None:
        cancer_type_df = pd.read_csv(os.path.join(self.processed_data_dir, "cancer_type.tsv"), sep="\t")
        if self.cfg["cancer_type"] != "all":
            cancer_type_sample_ids = cancer_type_df[cancer_type_df["cancer_type"] == self.cfg["cancer_type"]]["sample_id"].tolist()
        else:
            cancer_type_sample_ids = cancer_type_df["sample_id"].tolist()

        cancer_type_df = cancer_type_df[cancer_type_df["sample_id"].isin(cancer_type_sample_ids)]

        output_df = pd.read_csv(os.path.join(self.processed_data_dir, self.output_data_type + ".tsv"), sep="\t")
        output_df = output_df[output_df["sample_id"].isin(cancer_type_sample_ids)]

        thresholded_cna_mask_df = pd.read_csv(os.path.join(self.processed_data_dir, "thresholded_cna.tsv"), sep="\t")

        input_df = None

        for current_input_data_type in self.input_data_types:
            current_input_data_type_df = pd.read_csv(os.path.join(self.processed_data_dir, current_input_data_type + ".tsv"), sep="\t")

            if current_input_data_type in ["unthresholded_cna", "thresholded_cna", "rppa"]:
                intersecting_columns = set(current_input_data_type_df.columns).intersection(set(output_df.columns)).intersection(set(thresholded_cna_mask_df.columns))
                if self.cfg["gene_type"] == "all_genes":
                    pass
                elif self.cfg["gene_type"] in ["rppa_genes", "168_highly_expressed_genes", "1000_highly_expressed_genes"]:
                    intersecting_columns = set([str(entrezgene_id) for entrezgene_id in pd.read_csv(os.path.join(self.processed_data_dir, f"{self.cfg['gene_type']}.tsv"), sep="\t")["gene_id"].tolist()]).intersection(intersecting_columns)
                else:
                    raise Exception(f"{self.cfg['gene_type']} is not a valid gene_type.")

                if self.cfg["per_chromosome"]:
                    entrezgene_id_chromosome_name_mapping_df = pd.read_csv(os.path.join(self.processed_data_dir, "entrezgene_id_chromosome_name_mapping.tsv"), sep="\t")
                    entrezgene_id_chromosome_name_mapping_df = entrezgene_id_chromosome_name_mapping_df[entrezgene_id_chromosome_name_mapping_df["chromosome_name"].isin(["X", "Y"] + [str(i) for i in range(1, 23)])]
                    entrezgene_id_chromosome_name_mapping_df["entrezgene_id"] = entrezgene_id_chromosome_name_mapping_df["entrezgene_id"].apply(lambda x: str(x))
                    intersecting_columns = set(entrezgene_id_chromosome_name_mapping_df["entrezgene_id"].tolist()).intersection(intersecting_columns)
                    entrezgene_id_chromosome_name_mapping_df = entrezgene_id_chromosome_name_mapping_df[entrezgene_id_chromosome_name_mapping_df["entrezgene_id"].isin(intersecting_columns)]
                    chromosome_name_entrezgene_ids_mapping = dict(entrezgene_id_chromosome_name_mapping_df.groupby("chromosome_name")["entrezgene_id"].apply(list).reset_index(drop=False).values)

                intersecting_columns = ["sample_id"] + [column for column in sorted(intersecting_columns) if column != "sample_id"]

                output_df = output_df[intersecting_columns]
                current_input_data_type_df = current_input_data_type_df[intersecting_columns]
                thresholded_cna_mask_df = thresholded_cna_mask_df[intersecting_columns]

            current_input_data_type_df = current_input_data_type_df[current_input_data_type_df["sample_id"].isin(cancer_type_sample_ids)]

            if input_df is None:
                input_df = current_input_data_type_df
            else:
                input_df = pd.merge(left=input_df,
                                    right=current_input_data_type_df,
                                    on="sample_id",
                                    how="inner")

        if self.cfg["cancer_type"] == "all":
            cancer_type_one_hot_df = pd.read_csv(os.path.join(self.processed_data_dir, "cancer_type_one_hot.tsv"), sep="\t")
            input_df = pd.merge(left=input_df,
                                right=cancer_type_one_hot_df,
                                on="sample_id",
                                how="inner")

        # merge input, output and cancer_type dataframes
        merged_df = pd.merge(left=input_df, right=output_df, how="inner", on="sample_id")
        merged_df = pd.merge(left=merged_df, right=cancer_type_df, how="inner", on="sample_id")

        self.sample_ids = merged_df["sample_id"].values.ravel()
        self.sample_id_indices = np.arange(self.sample_ids.shape[0])
        merged_df.drop(columns=["sample_id"], inplace=True)

        self.one_hot_column_indices = [index for index, column in enumerate(merged_df.columns) if str(column).startswith("cancer_type_")]

        self.cfg["input_dimension"] = input_df.shape[1] - 1
        self.cfg["output_dimension"] = output_df.shape[1] - 1

        self.X = merged_df.iloc[:, :self.cfg["input_dimension"]].values
        self.y = merged_df.iloc[:, self.cfg["input_dimension"]:self.cfg["input_dimension"]+self.cfg["output_dimension"]].values
        self.all_cancer_types = merged_df["cancer_type"].values.ravel()

        self.logger.log(level=logging.INFO, msg=f"X.shape: {self.X.shape}, y.shape: {self.y.shape}")

        self.cfg["entrezgene_ids"] = [column for column in output_df.columns if column != "sample_id"]
        self.cfg["num_genes"] = len(self.cfg["entrezgene_ids"])

        if self.cfg["per_chromosome"]:
            self.cfg["chromosome_name_X_column_ids_mapping"] = {"nonchromosome": list(range(len(self.cfg["entrezgene_ids"]), self.X.shape[1], 1))}
            for chromosome_name, entrezgene_ids in chromosome_name_entrezgene_ids_mapping.items():
                self.cfg["chromosome_name_X_column_ids_mapping"][chromosome_name] = [np.argwhere(np.array(self.cfg["entrezgene_ids"]) == entrezgene_id).item() for entrezgene_id in entrezgene_ids]

        self.len_dataset = self.X.shape[0]

        all_stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=self.split_ratios["train"], random_state=self.seed)
        all_split_indices = next(all_stratified_shuffle_split.split(X=self.sample_id_indices, y=self.all_cancer_types))
        self.train_indices = all_split_indices[0]
        val_test_indices = all_split_indices[1]

        val_test_cancer_types = self.all_cancer_types[val_test_indices]
        val_test_stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=(self.split_ratios["val"] / (self.split_ratios["val"] + self.split_ratios["test"])), random_state=self.seed)
        val_test_split_indices = next(val_test_stratified_shuffle_split.split(X=val_test_indices, y=val_test_cancer_types))
        self.val_indices = val_test_indices[val_test_split_indices[0]]
        self.test_indices = val_test_indices[val_test_split_indices[1]]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.as_tensor(self.X[idx, :], device=self.device, dtype=torch.float32),
            "y": torch.as_tensor(self.y[idx, :], device=self.device, dtype=torch.float32),
            "sample_id_indices": torch.as_tensor(self.sample_id_indices[idx], dtype=torch.long)
        }

    def __len__(self) -> int:
        return self.len_dataset


class UnthresholdedCNAPurity2ZScoreDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["unthresholded_cna", "tumor_purity"],
            output_data_type="zscore",
            logger=logger
        )


class UnthresholdedCNA2ZScoreDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["unthresholded_cna"],
            output_data_type="zscore",
            logger=logger
        )


class ThresholdedCNAPurity2ZScoreDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["thresholded_cna", "tumor_purity"],
            output_data_type="zscore",
            logger=logger
        )


class ThresholdedCNA2ZScoreDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["thresholded_cna"],
            output_data_type="zscore",
            logger=logger
        )


class RPPA2ZScoreDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["rppa"],
            output_data_type="zscore",
            logger=logger
        )
