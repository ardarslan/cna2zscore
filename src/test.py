import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import pearsonr

from collections import defaultdict
import logging
from typing import Any, Dict, List, Tuple


def test(cfg: Dict[str, Any], data_loaders: List[DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32, np.float32, List[Tuple[int, np.float32]], List[Tuple[int, np.float32]]]:
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

        all_ground_truths = []
        all_predictions = []

        cna_ground_truths_1d = []
        cna_predictions_1d = []

        noncna_ground_truths_1d = []
        noncna_predictions_1d = []

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

            for i in range(cna_mask.shape[0]):
                for j in range(cna_mask.shape[1]):
                    if cna_mask[i][j] == 1.0:
                        cna_ground_truths_1d.append(float(y[i][j].cpu().numpy()))
                        cna_predictions_1d.append(float(yhat[i][j].cpu().numpy()))
                    else:
                        noncna_ground_truths_1d.append(float(y[i][j].cpu().numpy()))
                        noncna_predictions_1d.append(float(yhat[i][j].cpu().numpy()))

            all_ground_truths.append(y.cpu().numpy())
            all_predictions.append(yhat.cpu().numpy())

            all_count += float(y.shape[0] * y.shape[1])
            all_loss_sum += float(loss_function(yhat, y))

            cna_count += float(cna_mask.sum())
            cna_loss_sum += float(loss_function(yhat * cna_mask, y * cna_mask))

            noncna_count += float(noncna_mask.sum())
            noncna_loss_sum += float(loss_function(yhat * noncna_mask, y * noncna_mask))

            gene_counts += float(y.shape[0])

            for sample_id in range(y.shape[0]):
                for entrezgene_id, column_id in zip(entrezgene_ids, range(y.shape[1])):
                    gene_loss_sums[entrezgene_id] += float(loss_function(yhat[sample_id][column_id], y[sample_id][column_id]))

        all_ground_truths = np.vstack(all_ground_truths)
        all_predictions = np.vstack(all_predictions)

        all_ground_truths_1d = all_ground_truths.ravel()
        all_predictions_1d = all_predictions.ravel()

        cna_ground_truths_1d = np.array(cna_ground_truths_1d)
        cna_predictions_1d = np.array(cna_predictions_1d)

        noncna_ground_truths_1d = np.array(noncna_ground_truths_1d)
        noncna_predictions_1d = np.array(noncna_predictions_1d)

        all_corr, all_p_value = pearsonr(all_ground_truths_1d, all_predictions_1d)
        cna_corr, cna_p_value = pearsonr(cna_ground_truths_1d, cna_predictions_1d)
        noncna_corr, noncna_p_value = pearsonr(noncna_ground_truths_1d, noncna_predictions_1d)

        all_loss = all_loss_sum / all_count
        cna_loss = cna_loss_sum / cna_count
        noncna_loss = noncna_loss_sum / noncna_count
        gene_losses = dict(
            (entrezgene_id, gene_loss_sum / gene_counts) for entrezgene_id, gene_loss_sum in gene_loss_sums.items()
        )
        gene_losses = sorted(list(gene_losses.items()), key=lambda x: x[1])
        best_predicted_20_genes = gene_losses[:20]
        worst_predicted_20_genes = gene_losses[-20:]

        logger.log(level=logging.INFO, msg=f"   All genes, test {cfg['loss_function']} loss: {np.round(all_loss, 2)}.")
        logger.log(level=logging.INFO, msg=f"   CNA genes, test {cfg['loss_function']} loss: {np.round(cna_loss, 2)}.")
        logger.log(level=logging.INFO, msg=f"nonCNA genes, test {cfg['loss_function']} loss: {np.round(noncna_loss, 2)}.")

        logger.log(level=logging.INFO, msg=f"   All genes, test correlation: {np.round(all_corr, 2)} with p-value: {np.round(all_p_value, 2)}.")
        logger.log(level=logging.INFO, msg=f"   CNA genes, test correlation: {np.round(cna_corr, 2)} with p-value: {np.round(cna_p_value, 2)}.")
        logger.log(level=logging.INFO, msg=f"nonCNA genes, test correlation: {np.round(noncna_corr, 2)} with p-value: {np.round(noncna_p_value, 2)}.")

        logger.log(level=logging.INFO, msg=f"------------------------")

        logger.log(level=logging.INFO, msg=f"Best predicted 20 genes:")
        for entrezgene_id, gene_loss in best_predicted_20_genes:
            logger.log(level=logging.INFO, msg=f"Entrezgene ID: {entrezgene_id}, {cfg['loss_function']} loss: {np.round(gene_loss, 2)}.")
        logger.log(level=logging.INFO, msg=f"------------------------")

        logger.log(level=logging.INFO, msg=f"Worst predicted 20 genes:")
        for entrezgene_id, gene_loss in worst_predicted_20_genes:
            logger.log(level=logging.INFO, msg=f"Entrezgene ID: {entrezgene_id}, {cfg['loss_function']} loss: {np.round(gene_loss, 2)}.")

        return {
                "all_ground_truths": all_ground_truths,
                "all_predictions": all_predictions,
                "all_loss": all_loss,
                "all_corr": all_corr,
                "all_p_value": all_p_value,
                "cna_loss": cna_loss,
                "cna_corr": cna_corr,
                "cna_p_value": cna_p_value,
                "noncna_loss": noncna_loss,
                "noncna_corr": noncna_corr,
                "noncna_p_value": noncna_p_value,
                "best_predicted_20_genes": best_predicted_20_genes,
                "worst_predicted_20_genes": worst_predicted_20_genes
            }
