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

import tqdm
from typing import Dict
import numpy
import pandas
from trustfuse.conflicting_dataset.preprocessing import focus_on_attributes


class Model:
    """Generic Model"""

    def __init__(self, dataset, progress=tqdm, **kwargs):

        # output of the model
        self.truth_mapping = {}
        self.source_mapping = {}
        self.fused_data = ...
        self.source_nb = {}
        self.fact_nb = {}
        self.entity_names = {}
        self.attributes = {}
        self.claim_type = {}
        # input of the model Dict[bid: np.array]
        self.model_input = {}
        # output
        self.model_output = {}
        self.index = []
        self.fact = []
        self.numeric = False
        if "numeric" in kwargs:
            self.numeric = kwargs["numeric"]
        if self.numeric:
            focus_on_attributes(dataset, attributes = ["quantity"], action= "type_selection")
        self.transform_data(dataset, progress)


    def fuse(self, dataset, bid, inputs, progress=tqdm):
        """Perform the fusion

        Args:
            dataset (Dataset): Dataset instance

        Returns:
            Dict: Results with confidence scores
        """
        self._fuse(dataset, bid, inputs, progress)
        return self.get_results(dataset)


    def _fuse(self, dataset, bid, inputs, progress):
        raise NotImplementedError("Must be implemented in the child class")


    def transform_data(self, dataset, progress) -> Dict:
        """Method to comply with KDEm authors' implementation: 
        https://github.com/MengtingWan/KDEm/tree/master .
        Returns:
            (np.array, np.array, np.array): input of fusion models.
        """

        for bid in progress.tqdm(dataset.data.keys(), desc="Prepare data"):

            # Init the type for each set of conflicting facts
            self.claim_type[bid] = []
            # Construction of "data" truth_mapping =
            # {
            #     bid: {
            #         entity1: {
            #             attr1: 0
            #             attr2: 1,
            #             attr3: 2,
            #             ...
            #         }
            #         entity2: {
            #             attr1: 58,
            #             attr7: 89,
            #             ...
            #         }
            #     }
            # }
            self.truth_mapping[bid] = {}
            sources_names = list(set(dataset.data[bid]['Source']))
            # Assign an index for each source source_mapping =
            # {
            #     bid: {
            #         source1: 0,
            #         source2: 1,
            #         ...,
            #         sourceN: N
            #     }
            # }
            self.source_mapping[bid] = dict(
                zip(
                    sources_names,
                    list(range(len(sources_names)))
                    )
                )
            # If entity_col_name does not change over time ???
            # List of all entities present in the bucket bid
            # entity_names = [Q243, Q51, Q789, ..., QN] or
            # [flight#5, flight#7, ..., flight#N]
            self.entity_names[bid] = list(set(
                dataset.data[bid][dataset.entity_col_name]
                ))
            # List of all attributes
            data = []
            index_truth_mapping = 0
            # First we proceed by entity
            for entity in self.entity_names[bid]:
                # For truth reverse mapping after the fusion step
                self.truth_mapping[bid][entity] = {}
                # transformed_entity_data is an atomic element of claim (input of fusion models)
                # transformed_entity_data =
                # [
                #   [[index_source1, value],
                #   [index_source2, value],
                #  ...,
                #  [index_sourceN, value]],

                # [[index_source1, value]
                #  [index_source3, value]]
                #]
                transformed_entity_data = [[] for _ in dataset.attributes[bid]]
                bid_data = dataset.data[bid]
                # Select all attributes/values of a specific entity
                entity_data = bid_data[bid_data[dataset.entity_col_name] == entity]
                # Iterate over dataset tuples where each tuple has a specific Source
                # (attr1 -> val, attr2 -> val, ..., attrN -> val) | Source#K
                for _, row in entity_data.iterrows():
                    # Iterate over all the attributes of the tuple
                    for i, prop in enumerate(dataset.attributes[bid]):
                        # Check if the value provided by the source is not empty
                        if not pandas.isna(row[prop]):
                            # Get index of the source
                            source = self.get_source_index(bid, row['Source'])
                            value = row[prop]
                            transformed_entity_data[i].append([source, value])
                # Check if each attribute of the entity has at least one
                # value and define if this attribute is preserved in
                # the input of the fusion model for avoiding an empty list
                for sub_tab, attr in zip(
                    transformed_entity_data, dataset.attributes[bid]
                    ):
                    if len(sub_tab) > 0:
                        # For reverse mapping after the fusion step
                        self.truth_mapping[bid][entity][attr] = index_truth_mapping
                        if attr in dataset.attribute_types:
                            self.claim_type[bid] \
                            .append(dataset.attribute_types[attr])
                        else:
                            self.claim_type[bid] \
                            .append("string")
                        index_truth_mapping += 1

                data.extend([l for l in transformed_entity_data if len(l) > 0].copy())

            data = [numpy.array(ent, dtype=object) for ent in data]
            # To clean later
            # Number of sources
            self.source_nb[bid] = len(sources_names)
            # # entity_names x # attributes
            self.fact_nb[bid] = len(data)
            index = []
            # fact = claim
            fact = []
            count = numpy.zeros(self.source_nb[bid])
            for i in range(self.fact_nb[bid]):
                src = [val[0] for val in data[i]]
                src = [int(idx) for idx in src]
                count[src] = count[src] + 1
                index.append(src)
                fact.append(numpy.array([val[1] for val in data[i]],
                                        dtype=type(data[i][0][1])))

            self.model_input[bid] = [index, fact, count]


    def get_source_index(self, bid, source):
        return self.source_mapping[bid][source]


    def get_results(self, dataset) -> Dict:
        """Unify results for base algorithms

        Args:
            result (Dict): reconciliation model output dictionary
        """
        unified_result = {}
        for bid in self.model_output:
            fusion_result = self.model_output[bid]["truth"]
            source_weights = self.model_output[bid]["weights"]
            unified_result[bid] = {
                "truth": {},
                "weights": {}
            }
            for e in self.entity_names[bid]:
                if e not in unified_result[bid]["truth"]:
                    # Initialize the result for each attribute of the entity
                    unified_result[bid]["truth"][e] = {
                        attr: [None]
                        for attr in dataset.attributes[bid]
                        }
                for a in dataset.attributes[bid]:
                    # Get index of the corresponding attribute to retrieve
                    # the predicted value from de the fusion result
                    result_index = self.get_result_index(bid, e, a)
                    if result_index is not None:
                        values = fusion_result[result_index]
                        # Check if multiple predicted values
                        # are provided by the fusion model
                        if isinstance(values, list):
                            unified_result[bid]["truth"][e][a] = values
                        else:
                            unified_result[bid]["truth"][e][a] = [values]
            # Record source weights
            for source_name, source_index in self.source_mapping[bid].items():
                unified_result[bid]["weights"][source_name] = source_weights[source_index]

        return unified_result


    def get_result_index(self, bid, entity, attribute):
        if attribute in self.truth_mapping[bid][entity]:
            return self.truth_mapping[bid][entity][attribute]
        return None


    def convert_weights(self, bid, source_accuracy):
        source_weights = numpy.zeros(self.source_nb[bid])
        for ind, weight in source_accuracy.items():
            source_weights[ind] = weight
        return source_weights
