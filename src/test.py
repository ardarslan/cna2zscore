import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from collections import defaultdict
import logging
from typing import Any, Dict, List


def test(cfg: Dict[str, Any], data_loaders: List[DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> None:
    model.eval()

    with torch.no_grad():
        all_count = 0.0
        all_loss_sum = 0.0

        cna_count = 0.0
        cna_loss_sum = 0.0

        noncna_count = 0.0
        noncna_loss_sum = 0.0

        entrezgene_ids = dataset.entrezgene_ids
        gene_counts = 0.0
        gene_loss_sums = defaultdict(lambda: 0.0)

        test_ground_truths = []
        test_predictions = []

        for batch in data_loaders["test"]:
            X = batch["X"]
            y = batch["y"]
            cna_mask = batch["mask"]
            noncna_mask = 1.0 - cna_mask

            if cfg["normalize_input"]:
                X = (X - dataset.X_train_mean) / (dataset.X_train_std + 1e-10)

            yhat = model(X)

            if cfg["normalize_output"]:
                # then during training y was manually normalized, and yhat is produced as normalized as well.
                # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                yhat = yhat * (dataset.y_train_std + 1e-10) + dataset.y_train_mean

            test_ground_truths.append(y)
            test_predictions.append(yhat)

            all_count += y.shape[0] * y.shape[1]
            all_loss_sum += float(loss_function(yhat, y))

            cna_count += cna_mask.sum()
            cna_loss_sum += float(loss_function(yhat * cna_mask, y * cna_mask))

            noncna_count += noncna_mask.sum()
            noncna_loss_sum += float(loss_function(yhat * noncna_mask, y * noncna_mask))

            gene_counts += y.shape[0]

            for sample_id in range(y.shape[0]):
                for entrezgene_id, column_id in zip(entrezgene_ids, range(y.shape[1])):
                    gene_loss_sums[entrezgene_id] += loss_function(yhat[sample_id][column_id], y[sample_id][column_id])

        test_ground_truths = torch.vstack(test_ground_truths)
        test_predictions = torch.vstack(test_predictions)

        all_loss = np.round(all_loss_sum / all_count, 2)
        cna_loss = np.round(cna_loss_sum / cna_count, 2)
        noncna_loss = np.round(noncna_loss_sum / noncna_count, 2)
        gene_losses = dict(
            (entrezgene_id, np.round(np.float32(gene_loss_sum / gene_counts), 2)) for entrezgene_id, gene_loss_sum in gene_loss_sums.items()
        )
        gene_losses = sorted(list(gene_losses.items()), key=lambda x: x[1])
        best_predicted_20_genes = gene_losses[:20]
        worst_predicted_20_genes = gene_losses[-20:]

        logger.log(level=logging.INFO, msg=f"All genes, test {cfg['loss_function']} loss: {all_loss}.")
        logger.log(level=logging.INFO, msg=f"CNA genes, test {cfg['loss_function']} loss: {cna_loss}.")
        logger.log(level=logging.INFO, msg=f"nonCNA genes, test {cfg['loss_function']} loss: {noncna_loss}.")
        logger.log(level=logging.INFO, msg=f"------------------------")

        logger.log(level=logging.INFO, msg=f"Best predicted 20 genes:")
        for entrezgene_id, gene_loss in best_predicted_20_genes:
            logger.log(level=logging.INFO, msg=f"Entrezgene ID: {entrezgene_id}, {cfg['loss_function']} loss: {gene_loss}.")
        logger.log(level=logging.INFO, msg=f"------------------------")

        logger.log(level=logging.INFO, msg=f"Worst predicted 20 genes:")
        for entrezgene_id, gene_loss in worst_predicted_20_genes:
            logger.log(level=logging.INFO, msg=f"Entrezgene ID: {entrezgene_id}, {cfg['loss_function']} loss: {gene_loss}.")

        return test_ground_truths, test_predictions
