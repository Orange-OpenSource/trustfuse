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

import pandas as pd
import os
import tqdm
from typing import Dict, Tuple, Mapping
import pickle
import copy
import tabulate
import networkx
from trustfuse.conflicting_dataset.preprocessing import (DATA_PREPROCESSING_FUNCTIONS,
                                                         METADATA_PREPROCESSING_FUNCTIONS,
                                                         data_preprocessing)
from trustfuse.visualization.visualization import set_colors_map


def rename_data_header(df_dict, header):
	for k in df_dict:
		df_dict[k].rename(
			columns={i: header[i] for i in range(len(header))},
			inplace=True
			)


def complete_gt(prev_df, next_df, entity_name):

    row_from_prev_df = prev_df[prev_df["Entity"] == entity_name]

    if not row_from_prev_df.empty:
        data_dict = row_from_prev_df.iloc[0].to_dict()

        for col, value in data_dict.items():
            if col != "Entity":
                entity_df = next_df[next_df["Entity"] == entity_name]
                if col in entity_df:
                    if not isinstance(entity_df[col].iloc[0], list) \
                        and pd.isna(entity_df[col].iloc[0]):
                        next_df.loc[next_df["Entity"] == entity_name, col] = [value]
                else:
                    next_df.loc[next_df["Entity"] == entity_name, col] = [value]
        

class Dataset:
    def __init__(self, entity_col_name: str, attribute_types):

        # Column of the dataset that represents the entities
        self.entity_col_name = entity_col_name
        # Ground Truth Data
        self.gt_data = {}
        # Conflicting Data
        self.data = {}
        # List of attributes involved in each bucket
        self.attributes = {}
        # Dict mapping between attribute/datatype
        self.attribute_types = attribute_types
        # Make copies for reverse mapping after preprocessing
        self.seed_data = {}
        self.seed_gt_data = {}
        # Init formatted result variables
        self.fmt_fused_data = {}

        self.weights_dict = {}
        self.fused_data = {}
        # postpreprocess
        self.data_pp = {}
        self.nb_buckets = 0
        # Set of known entities for each bucket
        self.is_known = {}

        self.colors_map = None


    def compute_known_entities(self):
        """Compute known entities for each bucket after the preprocessing step"""
        self.is_known = {}
        for bid, _ in self.data.items():
            if bid == 0:
                self.is_known[bid] = set()
            else:
                df = self.data[bid-1]
                relevant_cols = [col for col in df.columns if col not in ["Source"]]
                values = df[relevant_cols].values.ravel()
                previous_bucket_entities = set(values)
                self.is_known[bid] = self.is_known[bid - 1] | previous_bucket_entities


    def __str__(self):
        """Print the attributes of the instance"""
        return ("\n".join(f"{cle}: {valeur}"
                          for cle, valeur in self.__dict__.items()))


    def set_attributes(self, attributes):
        """Attributes setter"""
        self.attributes = attributes


    def apply_preprocessors(self, preprocessors, modify_structure=False,
                            progress=tqdm, **kwargs):
        """Apply preprocessors"""
        for preprocessor in preprocessors:
            if preprocessor in DATA_PREPROCESSING_FUNCTIONS:
                data_preprocessing(
                    self,
                    DATA_PREPROCESSING_FUNCTIONS[preprocessor],
                    modify_structure=modify_structure,
                    progress=progress,
                    **preprocessors[preprocessor],
                    **kwargs
                    )


    def apply_data_preprocessing(self, preprocessors, progress=tqdm, **kwargs):
        """Apply preprocessing on data

        Args:
            preprocessors (Dict): Dict with preprocessors and its parameter
        """
        if "modify_structure" in preprocessors:
            self.apply_preprocessors(preprocessors["modify_structure"],
                                     modify_structure=True,
                                     progress=progress,
                                     **kwargs)
        # Copy data before the preprocess step in order to save data
        # before indexing it for Reverse Mapping to display result in
        # the original format. This copy will support metrics computation
        self.make_post_preprocess_copy()
        if "modify_data" in preprocessors:
            self.apply_preprocessors(preprocessors["modify_data"],
                                     progress=progress,
                                     **kwargs)


    def apply_metadata_preprocessors(self, preprocessors, modify_structure=False, progress=tqdm, **kwargs):
        """Apply metadata preprocessors"""
        for preprocessor in preprocessors:
            if preprocessor in METADATA_PREPROCESSING_FUNCTIONS:
                METADATA_PREPROCESSING_FUNCTIONS[preprocessor](self,
                                                               progress=progress,
                                                               **preprocessors[preprocessor],
                                                               **kwargs)


    def apply_metadata_preprocessing(self, preprocessors, progress=tqdm, **kwargs):
        """Apply preprocessing on data

        Args:
            preprocessors (Dict): Dict with preprocessors and its parameter
        """
        if "modify_structure" in preprocessors:
            self.apply_metadata_preprocessors(preprocessors["modify_structure"],
                                              modify_structure=True,
                                              progress=progress,
                                              **kwargs)
        if "modify_data" in preprocessors:
            self.apply_metadata_preprocessors(preprocessors["modify_data"],
                                              progress=progress,
                                              **kwargs)


    def serialize(self, path):
        """Serialize the dataset object and its state to save experiments"""
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    def make_post_preprocess_copy(self):
        """Make a copy of conflicting data after the preprocessing
        step used to compute metrics and allow the reverse mapping after 
        the fusion step
        """
        for bid, _ in self.data.items():
            self.data_pp[bid] = self.data[bid].copy(deep=True)
            # Make copies of the data + GT data before preprocessing
            # for reverse mapping aftre the fusion stage
            self.seed_data[bid] = self.data[bid].copy(deep=True)
            self.seed_gt_data[bid] = self.gt_data[bid].copy(deep=True)


    def reverse_mapping(self, unified_result: Mapping[str, Tuple], bid, progress=tqdm) \
        -> Tuple[Mapping[str, pd.DataFrame], Mapping[str, Dict]]:
        """Apply a reverse mapping to display fusion result
        as the same format of the input data"""
        # fusion_results: Tuple[np.array, np.array]
        # Iterate over the buckets
        # Creation of an "index" column as primary key
        # for mapping between seed and transformed data
        # Create an index column with the number
        # of each line in the Dataframe
        self.data[bid]["index"] = self.data[bid].index
        # Create the indexes for preprocessed
        # data on all attributes & entity column
        self.data[bid].set_index(
            self.attributes[bid] + [self.entity_col_name],
            inplace=True
            )

        self.seed_data[bid]["index"] = self.seed_data[bid].index
        self.seed_data[bid].set_index(
            [self.entity_col_name] + ["index"],
            inplace=True
            )

        # We retrieve the confidence scores of the sources
        fusion_result = unified_result[bid]["truth"]
        self.weights_dict[bid] = unified_result[bid]["weights"]
        # Create a dict to construct a Dataframe of the results
        self.fused_data[bid] = {a: [] for a in self.attributes[bid]}
        self.fused_data[bid][self.entity_col_name] = []
        # Make a copy of the dictionary
        self.fmt_fused_data[bid] = copy.deepcopy(self.fused_data[bid])

        for ent in progress.tqdm(fusion_result, desc="Reverse Mapping"):
            # Add the entity in the Entity column of the dict
            self.fused_data[bid][self.entity_col_name].append(ent)
            self.fmt_fused_data[bid][self.entity_col_name].append(ent)
            # Iterate over each attribute a of the entity e
            for a in fusion_result[ent]:
                # formatted list to fill in
                fmt_list = []
                # Iterate over the values of each property of the entity
                for v in fusion_result[ent][a]:
                    if v is not None:
                        # Firstly find the index where attribute == value from transformed dataset
                        subset = self.data[bid].xs(ent, level=self.entity_col_name)
                        corresponding_index = subset[
                            (subset.index.get_level_values(a) == v)
                        ]["index"].iloc[0]
                        # Then find the seed value with the corresponding index
                        seed_subset = self.seed_data[bid].xs(
                            ent,
                            level=self.entity_col_name
                        )
                        seed_value = seed_subset[
                            (seed_subset.index.get_level_values("index") == corresponding_index)
                        ][a].iloc[0]
                        fmt_list.append(seed_value)
                    else:
                        fmt_list.append(None)
                self.fused_data[bid][a].append(fusion_result[ent][a])
                self.fmt_fused_data[bid][a].append(fmt_list)
        self.fmt_fused_data[bid] = pd.DataFrame(data=self.fmt_fused_data[bid])
        self.fused_data[bid] = pd.DataFrame(data=self.fused_data[bid])

        return self.fused_data, self.weights_dict


    def print_table(self, metrics):
        for bid, result in self.fmt_fused_data.items():
            print(f"Bucket level: precision={round(metrics['buckets'][bid]['b_p'], 2)}  "
                f"recall={round(metrics['buckets'][bid]['b_r'], 2)}  "
                f"accuracy={round(metrics['buckets'][bid]['b_acc'], 2)}  "
                f"f1-score={round(metrics['buckets'][bid]['b_f1_score'], 2)} "
                f"completion rate={round(metrics['buckets'][bid]['c_rate'], 2)}")
            print(tabulate.tabulate(result.iloc[:100], headers='keys', tablefmt='grid'))


class StaticDataset(Dataset):
    """To handle literature datasets"""
    def __init__(self,
                 data_folder: str,
                 headers=None,
                 sep="\t",
                 gradio=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.headers = headers
        self.sep = sep
        if gradio:
            bucket_id = 0
            for file_name in tqdm.tqdm(data_folder[0]):
                self.data[bucket_id] = pd.read_csv(file_name,
                                                header=None,
                                                sep=sep,
                                                encoding='ISO-8859-1')
                self.data[bucket_id] = self.data[bucket_id] \
                    .drop(columns=[18], errors='ignore')
                bucket_id += 1
            bucket_id = 0
            for file_name in tqdm.tqdm(data_folder[1]):
                self.gt_data[bucket_id] = pd.read_csv(file_name,
                                                    header=None,
                                                    sep=sep,
                                                    encoding='ISO-8859-1')
                self.gt_data[bucket_id] = self.gt_data[bucket_id] \
                    .drop(columns=[17], errors='ignore')
                bucket_id += 1
        else:
            gt_data_path = os.path.join(data_folder, 'ground_truth')
            conflicting_data_path = os.path.join(data_folder, 'conflicting_data')
            bucket_id = 0
            for file_name in tqdm.tqdm(os.listdir(conflicting_data_path)):
                bucket_file = os.path.join(conflicting_data_path, file_name)
                self.data[bucket_id] = pd.read_csv(bucket_file,
                                                header=None,
                                                sep=sep,
                                                encoding='ISO-8859-1')
                self.data[bucket_id] = self.data[bucket_id] \
                    .drop(columns=[18], errors='ignore')
                bucket_id += 1
            bucket_id = 0
            for file_name in tqdm.tqdm(os.listdir(gt_data_path)):
                bucket_file = os.path.join(gt_data_path, file_name)
                self.gt_data[bucket_id] = pd.read_csv(bucket_file,
                                                    header=None,
                                                    sep=sep,
                                                    encoding='ISO-8859-1')
                self.gt_data[bucket_id] = self.gt_data[bucket_id] \
                    .drop(columns=[17], errors='ignore')
                bucket_id += 1

        # Define headers if they are not specified in the dataset
        if headers is not None:
            rename_data_header(self.data, headers[0])
            rename_data_header(self.gt_data, headers[1])

        # Define the attributes for each bucket
        for bid, _ in self.data.items():
            self.attributes[bid] = []
            self.attributes[bid].extend(list(self.data[bid].columns))
            self.attributes[bid].remove(self.entity_col_name)
            self.attributes[bid].remove('Source')
        self.colors_map = set_colors_map(self.attributes)


class DynamicDataset(Dataset):
    """Class to handle dynamic dataset (WikiConflict)"""
    def __init__(self, buckets_file, entity_as, **kwargs):
        super().__init__(**kwargs)

        with open(buckets_file, "rb") as f:
            buckets_by_qid = pickle.load(f)

        self.nb_buckets = len(buckets_by_qid)

        self.partial_orders = {}

        for bid in buckets_by_qid:
            # Load the right buckets
            if entity_as == "string":
                self.gt_data[bid] = buckets_by_qid[bid]["GT"]["value"]
                self.data[bid] = buckets_by_qid[bid]["data"]["value"]
            else:
                self.gt_data[bid] = buckets_by_qid[bid]["GT"]["qid"]
                self.data[bid] = buckets_by_qid[bid]["data"]["qid"]

                columns_to_keep = self.data[bid].columns.intersection(self.gt_data[bid].columns).tolist()
                columns_to_keep.append("Source")
                self.data[bid] = self.data[bid][columns_to_keep]

            # Each bucket must include the previous bucket
            if bid - 1 in self.data:
                # Conflicting data
                self.data[bid] = pd.concat([self.data[bid-1], self.data[bid]],
                                           ignore_index=True, sort=False)

                current_entity = self.gt_data[bid]["Entity"].iloc[0]
                df_fusion_mask = self.gt_data[bid-1][~self.gt_data[bid-1]["Entity"] \
                                                     .isin(self.gt_data[bid]["Entity"])]
                self.gt_data[bid] = pd.concat([df_fusion_mask, self.gt_data[bid]],
                                              ignore_index=True)

                complete_gt(self.gt_data[bid-1], self.gt_data[bid], current_entity)

                columns_to_keep = self.data[bid] \
                    .columns.intersection(self.gt_data[bid].columns).tolist()
                columns_to_keep.append("Source")
                self.data[bid] = self.data[bid][columns_to_keep]

                for attr, _ in buckets_by_qid[bid-1]["GT"]["value_order"].items():
                    if attr not in buckets_by_qid[bid]["GT"]["value_order"]:
                        buckets_by_qid[bid]["GT"]["value_order"][attr] = \
                            buckets_by_qid[bid-1]["GT"]["value_order"][attr]

            # Define the attributes for each bucket
            self.attributes[bid] = list(self.data[bid].columns)
            self.attributes[bid].remove(self.entity_col_name)
            self.attributes[bid].remove('Source')

        self.colors_map = set_colors_map(self.attributes)

        for bid in buckets_by_qid:
            self.partial_orders[bid] = self.create_partial_order_graphs(
                buckets_by_qid[bid]["GT"]["value_order"])


    def create_partial_order_graphs(self, partial_orders):
        """Generate the specificity partial orders in a non-binary tree.

        Args:
            partial_orders (dict): partials orders as nested lists
        """
        graphs = {}
        id = 0
        for attr, content in partial_orders.items():
            # Check if there is at least one partial order
            if len(content) > 0:
                graph = networkx.DiGraph()
                edges = []
                for partial_order in partial_orders[attr]:
                    max_depth = len(partial_order) - 1
                    roots = []
                    for root in partial_order[0]:
                        roots.append(id)
                        graph.add_node(id, label=root, depth=0, max_depth=max_depth,
                                       coeff=0 / max_depth, leaf=False, color=self.colors_map[attr])
                        id += 1
                    leaf = False
                    for depth, more_specific_values in enumerate(partial_order[1:]):
                        new_roots = []
                        if depth == len(partial_order[1:]) - 1:
                            leaf = True
                        for value in more_specific_values:
                            graph.add_node(id, label=value, depth=depth + 1, max_depth=max_depth,
                                           coeff=(depth + 1) / max_depth, leaf=leaf, color=self.colors_map[attr])
                            edges.extend([(id, root) for root in roots])
                            new_roots.append(id)
                            id += 1
                        roots = new_roots.copy()

                graph.add_edges_from(edges, label="more_specific_than")
                graphs[attr] = graph
        return graphs
