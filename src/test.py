import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def test(cfg: Dict[str, Any], data_loaders: List[DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32, np.float32, List[Tuple[int, np.float32]], List[Tuple[int, np.float32]]]:
    model.eval()

    all_sample_ids = []
    all_ys = []
    all_yhats = []
    all_cna_mask_nonbinaries = []
    all_cancer_types = []

    total_loss = 0.0
    total_sample_count = 0.0

    with torch.no_grad():
        for batch in data_loaders["test"]:
            sample_ids = batch["sample_id"]
            X = batch["X"]
            y = batch["y"]
            cna_mask_nonbinary = batch["mask"]
            cancer_types = batch["cancer_type"]

            if cfg["normalize_input"]:
                X = (X - dataset.X_train_mean) / dataset.X_train_std

            yhat = model(X)

            if cfg["normalize_output"]:
                # then during training y was manually normalized, and yhat is produced as normalized as well.
                # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                yhat = yhat * dataset.y_train_std + dataset.y_train_mean

            total_loss += float(loss_function(yhat, y))
            total_sample_count += X.shape[0]

            all_sample_ids.append(sample_ids.cpu().numpy().ravel())
            all_ys.append(y.cpu().numpy())
            all_yhats.append(yhat.cpu().numpy())
            all_cna_mask_nonbinaries.append(cna_mask_nonbinary.cpu().numpy())
            all_cancer_types.append(cancer_types.cpu().numpy().ravel())

    all_sample_ids = np.hstack(all_sample_ids)
    all_ys = np.vstack(all_ys)
    all_yhats = np.vstack(all_yhats)
    all_cna_mask_nonbinaries = np.vstack(all_cna_mask_nonbinaries)
    all_cancer_types = np.hstack(all_cancer_types)
    all_loss = total_loss / total_sample_count

    logger.log(level=logging.INFO, msg=f"Test {cfg['loss_function']} loss is {all_loss}.")

    test_results_dict = {
        "all_sample_ids": all_sample_ids,
        "all_ys": all_ys,
        "all_yhats": all_yhats,
        "all_cna_mask_nonbinaries": all_cna_mask_nonbinaries,
        "all_cancer_types": all_cancer_types
    }

    return test_results_dict



    # entrezgene_id_to_hgnc_symbol_mapping = get_entrezgene_id_to_hgnc_symbol_mapping(cfg=cfg)
    # zero = torch.tensor(data=0.0, device=cfg["device"])

    # with torch.no_grad():
    #     all_count = 0.0
    #     all_loss_sum = 0.0

    #     cna_count = 0.0
    #     cna_loss_sum = 0.0

    #     noncna_count = 0.0
    #     noncna_loss_sum = 0.0

    #     entrezgene_ids = dataset.entrezgene_ids
    #     gene_loss_sums = defaultdict(lambda: 0.0)
    #     gene_ground_truth_sums = defaultdict(lambda: 0.0)

    #     all_ground_truths = []
    #     all_predictions = []
    #     all_cna_mask_nonbinaries = []
    #     all_cancer_types = []

    #     cna_ground_truths_1d = []
    #     cna_predictions_1d = []

    #     noncna_ground_truths_1d = []
    #     noncna_predictions_1d = []

    #     for batch in data_loaders["test"]:
    #         X = batch["X"]
    #         y = batch["y"]
    #         cna_mask_nonbinary = batch["mask"]
    #         cna_mask_binary = batch["mask"].bool().float()
    #         cancer_types = batch["cancer_type"]

    #         noncna_mask_binary = 1.0 - cna_mask_binary

    #         if cfg["normalize_input"]:
    #             X = (X - dataset.X_train_mean) / dataset.X_train_std

    #         yhat = model(X)

    #         if cfg["normalize_output"]:
    #             # then during training y was manually normalized, and yhat is produced as normalized as well.
    #             # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
    #             yhat = yhat * dataset.y_train_std + dataset.y_train_mean

    #         # TODO: Write the for loops below in a more efficient way.
    #         for i in range(cna_mask_binary.shape[0]):
    #             for j in range(cna_mask_binary.shape[1]):
    #                 if cna_mask_binary[i][j] == 1.0:
    #                     cna_ground_truths_1d.append(float(y[i][j].cpu().numpy()))
    #                     cna_predictions_1d.append(float(yhat[i][j].cpu().numpy()))
    #                 else:
    #                     noncna_ground_truths_1d.append(float(y[i][j].cpu().numpy()))
    #                     noncna_predictions_1d.append(float(yhat[i][j].cpu().numpy()))

    #         all_ground_truths.append(y.cpu().numpy())
    #         all_predictions.append(yhat.cpu().numpy())
    #         all_cna_mask_nonbinaries.append(cna_mask_nonbinary.cpu().numpy())
    #         all_cancer_types.append(cancer_types.cpu().numpy())

    #         all_count += float(y.shape[0] * y.shape[1])
    #         all_loss_sum += float(loss_function(yhat, y))

    #         cna_count += float(cna_mask_binary.sum())
    #         cna_loss_sum += float(loss_function(yhat * cna_mask_binary, y * cna_mask_binary))

    #         noncna_count += float(noncna_mask_binary.sum())
    #         noncna_loss_sum += float(loss_function(yhat * noncna_mask_binary, y * noncna_mask_binary))

    #         for sample_id in range(y.shape[0]):
    #             for entrezgene_id, column_id in zip(entrezgene_ids, range(y.shape[1])):
    #                 gene_loss_sums[entrezgene_id] += float(loss_function(yhat[sample_id][column_id], y[sample_id][column_id]))
    #                 gene_ground_truth_sums[entrezgene_id] += float(loss_function(zero, y[sample_id][column_id]))

    #     all_ground_truths = np.vstack(all_ground_truths)
    #     all_predictions = np.vstack(all_predictions)
    #     all_cna_mask_nonbinaries = np.vstack(all_cna_mask_nonbinaries)
    #     all_cancer_types = np.vstack(all_cancer_types)

    #     all_ground_truths_1d = all_ground_truths.ravel()
    #     all_predictions_1d = all_predictions.ravel()

    #     cna_ground_truths_1d = np.array(cna_ground_truths_1d)
    #     cna_predictions_1d = np.array(cna_predictions_1d)

    #     noncna_ground_truths_1d = np.array(noncna_ground_truths_1d)
    #     noncna_predictions_1d = np.array(noncna_predictions_1d)

    #     all_corr, all_p_value = pearsonr(all_ground_truths_1d, all_predictions_1d)
    #     cna_corr, cna_p_value = pearsonr(cna_ground_truths_1d, cna_predictions_1d)
    #     noncna_corr, noncna_p_value = pearsonr(noncna_ground_truths_1d, noncna_predictions_1d)

    #     all_loss = all_loss_sum / all_count
    #     cna_loss = cna_loss_sum / cna_count
    #     noncna_loss = noncna_loss_sum / noncna_count
    #     gene_losses = dict(
    #         (entrezgene_id, gene_loss_sum / gene_ground_truth_sums[entrezgene_id]) for entrezgene_id, gene_loss_sum in gene_loss_sums.items()
    #     )
    #     gene_losses = sorted(list(gene_losses.items()), key=lambda x: x[1])
    #     best_predicted_20_gene_ids = gene_losses[:20]
    #     worst_predicted_20_gene_ids = gene_losses[-20:]

    #     best_predicted_20_gene_symbols = [(entrezgene_id_to_hgnc_symbol_mapping[int(entrezgene_id)], normalized_loss) for entrezgene_id, normalized_loss in best_predicted_20_gene_ids]
    #     worst_predicted_20_gene_symbols = [(entrezgene_id_to_hgnc_symbol_mapping[int(entrezgene_id)], normalized_loss) for entrezgene_id, normalized_loss in worst_predicted_20_gene_ids]

    #     logger.log(level=logging.INFO, msg=f"   All genes, test {cfg['loss_function']} loss: {np.round(all_loss, 2)}.")
    #     logger.log(level=logging.INFO, msg=f"   CNA genes, test {cfg['loss_function']} loss: {np.round(cna_loss, 2)}.")
    #     logger.log(level=logging.INFO, msg=f"nonCNA genes, test {cfg['loss_function']} loss: {np.round(noncna_loss, 2)}.")

    #     logger.log(level=logging.INFO, msg=f"   All genes, test correlation: {np.round(all_corr, 2)} with p-value: {np.round(all_p_value, 2)}.")
    #     logger.log(level=logging.INFO, msg=f"   CNA genes, test correlation: {np.round(cna_corr, 2)} with p-value: {np.round(cna_p_value, 2)}.")
    #     logger.log(level=logging.INFO, msg=f"nonCNA genes, test correlation: {np.round(noncna_corr, 2)} with p-value: {np.round(noncna_p_value, 2)}.")

    #     logger.log(level=logging.INFO, msg=f"------------------------")

    #     logger.log(level=logging.INFO, msg=f"Best predicted 20 genes:")
    #     for hgnc_symbol, _ in best_predicted_20_gene_symbols:
    #         logger.log(level=logging.INFO, msg=f"HGNC Symbol: {hgnc_symbol}")
    #     logger.log(level=logging.INFO, msg=f"------------------------")

    #     logger.log(level=logging.INFO, msg=f"Worst predicted 20 genes:")
    #     for hgnc_symbol, _ in worst_predicted_20_gene_symbols:
    #         logger.log(level=logging.INFO, msg=f"HGNC Symbol: {hgnc_symbol}")

    #     return {
    #             "all_ground_truths": all_ground_truths,
    #             "all_predictions": all_predictions,
    #             "all_cna_mask_nonbinaries": all_cna_mask_nonbinaries,
    #             "all_cancer_types": all_cancer_types,
    #             "all_loss": all_loss,
    #             "all_corr": all_corr,
    #             "all_p_value": all_p_value,
    #             "cna_loss": cna_loss,
    #             "cna_corr": cna_corr,
    #             "cna_p_value": cna_p_value,
    #             "noncna_loss": noncna_loss,
    #             "noncna_corr": noncna_corr,
    #             "noncna_p_value": noncna_p_value,
    #             "best_predicted_20_genes": best_predicted_20_gene_symbols,
    #             "worst_predicted_20_genes": worst_predicted_20_gene_symbols
    #         }
