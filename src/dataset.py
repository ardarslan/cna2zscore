import os
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        input_data_types: List[str],
        output_data_type: str,
        mask_data_type: Optional[str],
        logger: logging.Logger
    ):
        super().__init__()
        self.cfg = cfg
        self.input_data_types = input_data_types
        self.output_data_type = output_data_type
        self.mask_data_type = mask_data_type
        self.logger = logger

        if cfg["cancer_type"] == "all":
            self.cancer_types = ["blca", "lusc", "ov"]
        else:
            self.cancer_types = [cfg["cancer_type"]]

        self.one_hot_input_df = len(self.cancer_types) > 1
        self.split_ratios = cfg["split_ratios"]

        if "train" not in self.split_ratios.keys() or "val" not in self.split_ratios.keys() or "test" not in self.split_ratios.keys():
            raise Exception("The dictionary 'split_ratios' should have the following keys: 'train', 'val' and 'test'.")

        if self.split_ratios["train"] + self.split_ratios["val"] + self.split_ratios["test"] != 1.0:
            raise Exception("The values of dictionary 'split_ratios' should sum up to 1.0.")

        self.processed_data_dir = cfg["processed_data_dir"]
        self.seed = cfg["seed"]
        self.logger = logger

        self._process_dataset()
        self.input_dimension = self.X.shape[1]
        self.output_dimension = self.y.shape[1]

        self.device = cfg["device"]
        self.len_dataset = self.X.shape[0]

        all_indices = shuffle(range(self.len_dataset), random_state=self.seed)
        train_size = int(self.len_dataset * self.split_ratios["train"])
        val_size = int(self.len_dataset * self.split_ratios["val"])

        self.train_idx = all_indices[:train_size]
        self.val_idx = all_indices[train_size:train_size+val_size]
        self.test_idx = all_indices[train_size+val_size:]

        X_train = self.X[self.train_idx, :]
        y_train = self.y[self.train_idx, :]

        self.X_train_mean = torch.as_tensor(np.mean(X_train, axis=0), device=self.device)
        self.X_train_std = torch.as_tensor(np.std(X_train, axis=0), device=self.device)

        self.y_train_mean = torch.as_tensor(np.mean(y_train, axis=0), device=self.device)
        self.y_train_std = torch.as_tensor(np.std(y_train, axis=0), device=self.device)

    def _process_dataset(self) -> None:
        # prepare output dataframe
        output_df = pd.concat([pd.read_csv(os.path.join(self.processed_data_dir, cancer_type, self.output_data_type + ".tsv"), sep="\t") for cancer_type in self.cancer_types], axis=0)

        # prepare mask dataframe
        mask_df = pd.concat([pd.read_csv(os.path.join(self.processed_data_dir, cancer_type, self.mask_data_type + ".tsv"), sep="\t") for cancer_type in self.cancer_types], axis=0)

        # prepare input dataframe
        input_df = []
        for cancer_type in self.cancer_types:
            current_cancer_type_df = None

            for current_input_data_type in self.input_data_types:
                current_cancer_type_input_data_type_df = pd.read_csv(os.path.join(self.processed_data_dir, cancer_type, current_input_data_type + ".tsv"), sep="\t")

                if current_input_data_type == "cna":
                    intersecting_columns = sorted(list(set(current_cancer_type_input_data_type_df.columns).intersection(set(output_df.columns)).intersection(set(mask_df.columns))))
                    intersecting_columns = ["sample_id"] + [column for column in intersecting_columns if column != "sample_id"]
                    current_cancer_type_input_data_type_df = current_cancer_type_input_data_type_df[intersecting_columns]
                    mask_df = mask_df[intersecting_columns]
                    output_df = output_df[intersecting_columns]

                if current_cancer_type_df is None:
                    current_cancer_type_df = current_cancer_type_input_data_type_df
                else:
                    current_cancer_type_df = pd.merge(
                        left=current_cancer_type_df,
                        right=current_cancer_type_input_data_type_df,
                        on="sample_id",
                        how="inner"
                    )

            # add one hot encoding to the input dataframe if there are more than one cancer types
            if self.one_hot_input_df:
                current_cancer_type_df["cancer_type"] = cancer_type

            input_df.append(current_cancer_type_df)

        input_df = pd.concat(input_df, axis=0)

        # add one hot encoding to the input dataframe if there are more than one cancer types
        if self.one_hot_input_df:
            input_df = pd.get_dummies(input_df, columns=["cancer_type"])

        # drop input features with std 0.0
        input_features_with_0_std = input_df.drop(columns=["sample_id"]).loc[:, input_df.drop(columns=["sample_id"]).std(axis=0) == 0].columns.tolist()
        input_df = input_df[[column for column in input_df.columns if column not in input_features_with_0_std]]
        self.logger.log(level=logging.INFO, msg=f"Dropped the following input features with 0 std: {input_features_with_0_std}.")
        self.logger.log(level=logging.INFO, msg="Input features: " + ", ".join([column for column in input_df.columns]))

        if output_df.columns.tolist() != mask_df.columns.tolist():
            raise Exception("Columns of output_df should be the same with columns of mask_df.")

        # merge input, output and mask dataframes
        merged_df = pd.merge(left=input_df, right=output_df, how="inner", on="sample_id")
        merged_df = pd.merge(left=merged_df, right=mask_df, how="inner", on="sample_id")
        merged_df = merged_df.drop(columns=["sample_id"])
        merged_df = shuffle(merged_df, random_state=self.seed)

        self.input_dimension = input_df.shape[1] - 1
        self.output_dimension = output_df.shape[1] - 1

        self.X = merged_df.values[:, :self.input_dimension]
        self.y = merged_df.values[:, self.input_dimension:self.input_dimension + self.output_dimension]
        self.mask = merged_df.values[:, self.input_dimension + self.output_dimension:].astype(np.bool_).astype(np.float32)
        self.entrezgene_ids = [column for column in output_df.columns if column != "sample_id"]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
                "X": torch.as_tensor(self.X[idx, :], device=self.device, dtype=torch.float32),
                "y": torch.as_tensor(self.y[idx, :], device=self.device, dtype=torch.float32),
                "mask": torch.as_tensor(self.mask[idx, :], device=self.device, dtype=torch.float32)
               }

    def __len__(self) -> int:
        return self.len_dataset

    # def _process_dataset(self) -> None:
    #     # prepare output dataframe
    #     output_df = pd.concat([pd.read_csv(os.path.join(self.processed_data_dir, cancer_type, self.output_data_type + ".tsv"), sep="\t") for cancer_type in self.cancer_types], axis=0)

    #     # prepare input dataframe
    #     input_df = []
    #     for cancer_type in self.cancer_types:
    #         current_cancer_type_df = None

    #         for current_input_data_type in zip(self.input_data_types):
    #             current_cancer_type_input_data_type_df = pd.read_csv(os.path.join(self.processed_data_dir, cancer_type, current_input_data_type + ".tsv"), sep="\t")

    #             if current_cancer_type_df is None:
    #                 current_cancer_type_df = current_cancer_type_input_data_type_df
    #             else:
    #                 current_cancer_type_df = pd.merge(
    #                     left=current_cancer_type_df,
    #                     right=current_cancer_type_input_data_type_df,
    #                     on="sample_id",
    #                     how="inner"
    #                 )

    #         # add one hot encoding to the input dataframe if there are more than one cancer types
    #         if self.one_hot_input_df:
    #             current_cancer_type_df["cancer_type"] = cancer_type

    #         input_df.append(current_cancer_type_df)

    #     input_df = pd.concat(input_df, axis=0)

    #     # add one hot encoding to the input dataframe if there are more than one cancer types
    #     if self.one_hot_input_df:
    #         input_df = pd.get_dummies(input_df, columns=["cancer_type"])

    #     # drop input features with std 0.0
    #     input_features_with_0_std = input_df.drop(columns=["sample_id"]).loc[:, input_df.drop(columns=["sample_id"]).std(axis=0) == 0].columns.tolist()
    #     input_df = input_df[[column for column in input_df.columns if column not in input_features_with_0_std]]
    #     self.logger.log(level=logging.INFO, msg=f"Dropped the following input features with 0 std: {input_features_with_0_std}.")
    #     self.logger.log(level=logging.INFO, msg="Input features: " + ", ".join([column for column in input_df.columns]))

    #     # merge input, output and mask dataframes
    #     merged_df = pd.merge(left=input_df, right=output_df, how="inner", on="sample_id")
    #     merged_df = merged_df.drop(columns=["sample_id"])
    #     merged_df = shuffle(merged_df, random_state=self.seed)

    #     self.input_dimension = input_df.shape[1] - 1
    #     self.output_dimension = output_df.shape[1] - 1

    #     self.X = merged_df.values[:, :self.input_dimension]
    #     self.y = merged_df.values[:, self.input_dimension:self.input_dimension + self.output_dimension]

    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     return {
    #             "X": torch.as_tensor(self.X[idx, :], device=self.device, dtype=torch.float32),
    #             "y": torch.as_tensor(self.y[idx, :], device=self.device, dtype=torch.float32)
    #            }


class CNAPurity2GEXDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["cna", "purity"],
            output_data_type="gex",
            mask_data_type="cna_thresholded",
            logger=logger
        )


class RPPA2GEXDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["rppa"],
            output_data_type="gex",
            mask_data_type="cna_thresholded",
            logger=logger
        )


class AverageGEXSubtype2GEXDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        super().__init__(
            cfg=cfg,
            input_data_types=["avg_gex", "subtype"],
            output_data_type="gex",
            mask_data_type="cna_thresholded",
            logger=logger
        )
