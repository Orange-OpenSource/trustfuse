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

from __future__ import division
import argparse
import pickle
from trustfuse.conflicting_dataset.dataset import (DynamicDataset,
                                                   StaticDataset)
from trustfuse.evaluation import evaluation
import json
import settings
import logging
import tqdm

logging.basicConfig(level=logging.INFO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path",
                        dest='dataset_path',
                        help="Path to the dataset containing conflicting data",
                        required=True)
    parser.add_argument("--dataset-name",
                        dest='dataset_name',
                        help="Name of the dataset",
                        type=str,
                        choices=["Flight", "Stock", "Book", "WikiConflict"],
                        required=True)
    parser.add_argument('--attr-types',
                        dest='attr_types',
                        required=True)
    parser.add_argument('--buckets-file',
                        dest='buckets_file',
                        required=False)
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        choices=["CRH", "CATD", "KDEm", "GTM",
                                 "TruthFinder", "TKGC", "SLIMFAST", 
                                 "LTM", "ACCU"])
    parser.add_argument('--preprocess-config',
                        dest='preprocess_config',
                        required=False)
    parser.add_argument('--dynamic',
                        action='store_true')
    args = parser.parse_args()


    if "json" in args.attr_types:
        with open(args.attr_types, encoding="utf-8") as f:
            attr_types = json.load(f)
    else:
        with open(args.attr_types, 'rb') as f:
            attr_types = pickle.load(f)
            # Define types not in configuration file
            attr_types["label_en"] = "string"
            attr_types["label_fr"] = "string"
            attr_types["description_en"] = "string"
            attr_types["description_fr"] = "string"

    dataset = None
    if args.dynamic:
        dataset = DynamicDataset(args.buckets_file,
                                    attribute_types=attr_types,
                                    **settings.DATASET_PARAMETERS[args.dataset_name],
                                    entity_as="string")
        dataset.make_post_preprocess_copy()
    else:
        dataset = StaticDataset(args.dataset_path,
                                attribute_types=attr_types,
                                **settings.DATASET_PARAMETERS[args.dataset_name])

    # Preprocess
    if args.preprocess_config:
        with open(args.preprocess_config, encoding="utf-8") as preprocessing_config_file:
            preprocess_config = json.load(preprocessing_config_file)
        logging.info("DATA PREPROCESSING")
        dataset.apply_data_preprocessing(preprocess_config)
        logging.info("METADATA PREPROCESSING")
        dataset.apply_metadata_preprocessing(preprocess_config)
        logging.info("PREPROCESSING COMPLETED")

    # Load model
    model = settings.MODEL_MAP[args.model](dataset, **settings.MODEL_PARAMETERS[args.model])

    # Perform fusion
    for bid, inputs in tqdm.tqdm(model.model_input.items(), desc="Fusion"):
        results = model.fuse(dataset, bid, inputs)
        # Apply reverse mapping and record the fusion results in the Dataset object
        dataset.reverse_mapping(results, bid)

    metrics, _ = evaluation.get_metrics(dataset, dataset.attributes)

    metrics_to_print = ["ov_p", "ov_r", "ov_acc", "ov_f1_score", "specificity"]

    metrics_to_print = {key: metrics[key] for key in metrics_to_print if key in metrics}
    print(metrics_to_print)


if __name__ == "__main__":
    main()
