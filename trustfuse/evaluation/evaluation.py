"""
Software Name : TrustFuse
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
 
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
 
Authors: Lucas Jarnac, Yoan Chabot, and Miguel Couceiro
Software description: TrustFuse is a testbed that supports experimentation with data fusion models,
their evaluation, and the visualization of datasets as graphs or tables within a unified user interface.
"""

from typing import Set, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import tqdm

from trustfuse.conflicting_dataset.dataset import Dataset


def get_specificity_value(partial_orders, it_attr, attr, wrong_values, mode="negative"):
    """Compute the specificity value for each predicted value

    Args:
        partial_orders (Dict): Dict mapping each attribute to a partial order graph
        it_attr (Set): Set of 2-tuples of predicted values 
        attr (str): attribute for which the metric is computed
        wrong_values (Set): Set of 2-tuples containing all wrong predicted values
        mode (str): Consider incorrect values as negative coefficient or ignore them

    Returns:
        float: The specificity metric for the attribute
    """
    specificity = 0
    attr_ordered_entities_tot = 0

    # Remove wrong values to search the positive specificity coefficient
    correct_values = it_attr - wrong_values
    # Assigne negative specificity coeeficient to wrong values
    if mode == "negative":
        specificity = -1 * len(wrong_values)

    for result in correct_values:
        if attr in partial_orders and result[1] in partial_orders[attr]:
            specificity += partial_orders[attr]._node[result[1]]["coeff"]
            attr_ordered_entities_tot += 1

    specificity_value = 0
    if attr_ordered_entities_tot > 0:
        specificity_value = specificity / attr_ordered_entities_tot
    else:
        specificity_value = None
        specificity = None

    return specificity, specificity_value, attr_ordered_entities_tot


def gradient_prec(p):
    """Generate a color between red and green for performance visualization"""
    if pd.isna(p):
        return ""
    red = int(255 * (1 - p))
    green = int(255 * p)
    precision = p * 100
    return f"\033[97m\033[48;2;{red};{green};0m{precision:.2f}%\033[0m"


def compute_precision(tp, fp):
    if fp == 0:
        return 1
    return tp / (tp + fp)


def compute_recall(tp, fn):
    if (tp + fn) > 0:
        return tp / (tp + fn)
    return 0


def compute_accuracy(tp, fp, tn, fn):
    tot = tp + tn + fp + fn
    if tot > 0:
        return (tp + tn) / tot
    return 0


def compute_f1_score(r, p):
    if (r + p) > 0:
        return 2 * (r * p) / (r + p)
    return 0


def add_metrics(dataset, metrics):
    for bid in metrics["buckets"]:
        new_names = {}
        for attr in metrics["buckets"][bid]["attributes"]:
            new_name = attr
            for metric in metrics["buckets"][bid]["attributes"][attr]:
                new_name += f"\n{metric}=" \
                    f"{gradient_prec(metrics['buckets'][bid]['attributes'][attr][metric])}"
            new_names[attr] = new_name
        dataset.fmt_fused_data[bid].rename(columns=new_names, inplace=True)


def col2tuples(df: pd.DataFrame, entity_name: str, attr: str) -> Set[Tuple]:
    tuple_set = set()
    for _, row in df[[entity_name, attr]].iterrows():
        if isinstance(row[attr], list):
            for value in row[attr]:
                if value is not None:
                    tuple_set.add((row[entity_name], value))
        else:
            if not pd.isna(row[attr]) and row[attr] is not None:
                tuple_set.add((row[entity_name], row[attr]))
    return tuple_set


def get_metrics(dataset: Dataset, attributes=None, progress=tqdm, **kwargs) -> float:
    """Compute error rate & precision of the model on 
    the set attributes over all buckets in attributes.keys()

    Args:
        dataset (Dataset): dataset
        truth_scores_dict (Mapping[str, pd.DataFrame]): The truth scores 
        (predictions) outputted by the model
        attributes (Dict): attributes dictionary that contains 
        attributes to evaluate for each bucket

    Returns:
        float: (error_rate, precision)
    """

    # AH  = Average Hierarchical Score

    if attributes is None:
        attributes = dataset.attributes

    metrics = {
        "buckets": {
            bid: {
                "attributes": {
                    attr: {
                        "p": 0,
                        "r": 0,
                        "acc": 0,
                        "f1_score": 0,
                        "c_rate": 0,
                        "specificity": 0
                    } for attr in attributes[bid]
                }
            } for bid in dataset.fmt_fused_data.keys()
        }
    }

    metrics.update({
        "ov_p": 0,
        "ov_r": 0,
        "ov_acc": 0,
        "ov_f1_score": 0,
        "c_rate": 0,
        "specificity": 0
    })
    ov_tp, ov_fp, ov_tn, ov_fn = (0, 0, 0, 0)

    entity_col = dataset.entity_col_name

    dataset_specificity = 0
    dataset_ordered_entities_tot = 0

    for bid in progress.tqdm(dataset.fmt_fused_data,
                             desc="Metrics computation"):

        metrics["buckets"][bid].update({
            "b_p": 0,
            "b_r": 0,
            "b_acc": 0,
            "b_f1_score": 0,
            "c_rate": 0,
            "specificity": 0
        })
        b_tp, b_fp, b_tn, b_fn = (0, 0, 0, 0)

        bucket_specificity = 0
        bucket_ordered_entities_tot = 0

        # Ground Truth
        gt_df = dataset.seed_gt_data[bid]
        entities_in_gt = set(gt_df[entity_col])
        # Inferred Truth
        it_df = dataset.fmt_fused_data[bid][dataset.fmt_fused_data[bid][entity_col].isin(entities_in_gt)]

        # Data Post Preprocess
        dpp_df = dataset.data_pp[bid][dataset.data_pp[bid][entity_col].isin(entities_in_gt)]

        for attr in attributes[bid]:

            gt_attr = col2tuples(gt_df, entity_col, attr)
            it_attr = col2tuples(it_df, entity_col, attr)
            dpp_attr = col2tuples(dpp_df, entity_col, attr)

            # all_most_specific_values = set()

            # if attr in dataset.partial_orders[bid]:
            #     all_most_specific_values |= get_the_most_specific_values(
            #         dataset.partial_orders[bid][attr],
            #         "Q243")

            tp = len(gt_attr & it_attr)
            fp = len(it_attr - gt_attr)
            tn = len((dpp_attr - gt_attr) - it_attr)
            fn = len((dpp_attr - it_attr) & gt_attr)

            attr_specificity = None
            specificity_value = None

            if hasattr(dataset, "partial_orders") and attr in dataset.partial_orders[bid]:
                specificity_value, attr_specificity, tot_values = get_specificity_value(dataset.partial_orders[bid],
                                                                                        it_attr, attr,
                                                                                        it_attr-gt_attr, **kwargs)

                if specificity_value is not None:
                    bucket_specificity += specificity_value
                    bucket_ordered_entities_tot += tot_values
                    dataset_specificity += specificity_value
                    dataset_ordered_entities_tot += tot_values

            ov_tp += tp
            ov_fp += fp
            ov_tn += tn
            ov_fn += fn

            b_tp += tp
            b_fp += fp
            b_tn += tn
            b_fn += fn

            if len(gt_attr) > 0:
                c_rate = len(it_attr & gt_attr) / len(gt_attr)
            else:
                c_rate = 0
            metrics["buckets"][bid]["attributes"][attr]["c_rate"] = c_rate

            metrics["buckets"][bid]["c_rate"] += c_rate

            p = compute_precision(tp, fp)
            r = compute_recall(tp, fn)
            acc = compute_accuracy(tp, fp, tn, fn)

            metrics["buckets"][bid]["attributes"][attr].update({
                "p": p,
                "r": r,
                "acc": acc,
                "f1_score": compute_f1_score(r, p),
                "specificity": attr_specificity
            })

        attributes_count = len(metrics["buckets"][bid]["attributes"])
        if attributes_count > 0:
            metrics["buckets"][bid]["c_rate"] /= attributes_count

        b_p = compute_precision(b_tp, b_fp)
        b_r = compute_recall(b_tp, b_fn)
        b_acc = compute_accuracy(b_tp, b_fp, b_tn, b_fn)

        if bucket_ordered_entities_tot > 0:
            bucket_specificity /= bucket_ordered_entities_tot
        else:
            bucket_specificity = None

        metrics["buckets"][bid].update({
            "b_p": b_p,
            "b_r": b_r,
            "b_acc": b_acc,
            "b_f1_score": compute_f1_score(b_r, b_p),
            "specificity": bucket_specificity
        })

    metrics["c_rate"] /= len(metrics["buckets"])

    ov_p = compute_precision(ov_tp, ov_fp)
    ov_r = compute_recall(ov_tp, ov_fn)
    ov_acc = compute_accuracy(ov_tp, ov_fp, ov_tn, ov_fn)

    if dataset_ordered_entities_tot > 0:
        dataset_specificity /= dataset_ordered_entities_tot
    else:
        dataset_specificity = None

    metrics.update({
        "ov_p": ov_p,
        "ov_r": ov_r,
        "ov_acc": ov_acc,
        "ov_f1_score": compute_f1_score(ov_r, ov_p),
        "specificity": dataset_specificity
    })

    b_p_values = [metrics["buckets"][bid]['b_p'] for bid in metrics["buckets"]]
    metrics.update({
        "ov_p_median": np.median(b_p_values),
        "ov_p_var": np.var(b_p_values),
        "ov_p_avg": np.mean(b_p_values)
    })

    b_r_values = [metrics["buckets"][bid]['b_r'] for bid in metrics["buckets"]]
    metrics.update({
        "ov_r_median": np.median(b_r_values),
        "ov_r_var": np.var(b_r_values),
        "ov_r_avg": np.mean(b_r_values)
    })

    # Prepare data for gradio visualization
    dataset_avh = []
    for bid, _ in metrics["buckets"].items():
        if metrics["buckets"][bid]["specificity"] is not None:
            dataset_avh.append(round(metrics["buckets"][bid]["specificity"], 2))
        else:
            dataset_avh.append(0)
    if metrics["specificity"] is not None:
        dataset_avh.append(round(metrics["specificity"], 2))
    else:
        dataset_avh.append(0)

    tot_buckets = metrics["buckets"].keys()
    metrics_gr = {
        "Bucket": [f"Bucket#{i}" for i in metrics["buckets"].keys()] + ["Dataset"],
        "Precision": [round(metrics["buckets"][bid]["b_p"], 2)
                      for bid in tot_buckets]
                      + [round(metrics["ov_p"], 2)],
        "Recall": [round(metrics["buckets"][bid]["b_r"], 2)
                   for bid in tot_buckets]
                   + [round(metrics["ov_r"], 2)],
        "F1-score": [round(metrics["buckets"][bid]["b_f1_score"], 2)
                     for bid in tot_buckets]
                     + [round(metrics["ov_f1_score"], 2)],
        "Accuracy": [round(metrics["buckets"][bid]["b_acc"], 2)
                     for bid in tot_buckets]
                     + [round(metrics["ov_acc"], 2)],
        # "Completion rate": [round(metrics["buckets"][bid]["c_rate"], 2)
        #                     for bid in tot_buckets]
        #                     + [round(metrics["c_rate"], 2)],
        "Average hierarchical score": dataset_avh
    }

    metrics_gr = pd.DataFrame(metrics_gr)

    return metrics, metrics_gr


# CRH: for continuous data and measures the overall absolute
# distance from each method’s output to the ground truths
# attribute parameter = filter in the future
def compute_mnad(dataset: Dataset) -> float:
    """Compute the Mean Normalized Absolute Distance (MNAD)"""

    # Precision
    mae = 0
    tot = 0
    inferred_values = dataset.fused_data
    for bid, truth_scores in inferred_values.items():
        gt = dataset.gt_data[bid]
        entity_name = dataset.entity_col_name
        # Merging to align labeled data and model predictions
        common_entities = pd.merge(gt, truth_scores,
                                   on=entity_name, suffixes=('_gt', '_ts'))
        for attr in dataset.attributes[bid]:
            if dataset.attribute_types[attr] == "quantity":
                common_entities_clean = common_entities.dropna(
                    subset=[attr + '_gt', attr + '_ts'])
                gt_arr = common_entities_clean[attr + '_gt'].to_numpy()
                ts_arr = common_entities_clean[attr + '_ts'].to_numpy()
                if len(gt_arr) and len(ts_arr) > 0:
                    ts_arr = ts_arr[0]
                    # Normalization factor (variance)
                    var = np.var(abs(gt_arr - ts_arr))
                    if var == 0:
                        mae += mean_absolute_error(gt_arr, ts_arr)
                    else:
                        mae += mean_absolute_error(gt_arr, ts_arr) / var
                    tot += 1
    if tot == 0:
        return None
    return mae / tot
