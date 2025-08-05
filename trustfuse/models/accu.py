import random
import math
import numpy as np
from trustfuse.models.model import Model


class Accu(Model):
    def __init__(self, dataset, max_itr=100, **kwargs):

        super().__init__(dataset, **kwargs)
        self.itr = max_itr


    def _fuse(self, dataset, bid, inputs, progress):

        index, claim, _ = inputs

        source_observations = self.convert_input(index, claim)
        # load data
        object_truth = {}
        # init local dictionaries
        object_inferred_truth = {}
        object_distinct_observations = {}
        object_observations = {}
        # object to value dictionary
        for source_id, _ in source_observations.items():
            for oid in source_observations[source_id]:
                if oid in object_truth:
                    continue
                if oid not in object_observations:
                    object_observations[oid] = []
                    object_distinct_observations[oid] = set([])
                    object_inferred_truth[oid] = source_observations[source_id][oid]
                object_observations[oid].append((source_id, source_observations[source_id][oid]))
                object_distinct_observations[oid].add(source_observations[source_id][oid])
        # initialize source accuracy --- utilize any ground truth if specified
        source_accuracy = {}
        self._init_src_accuracy(source_observations,
                        object_truth, object_inferred_truth,
                        source_accuracy)

        for _ in range(self.itr):
            self.update_object_assignment(object_observations,
                                            object_distinct_observations,
                                            source_accuracy, object_inferred_truth)
            self.update_source_accuracy(source_observations,
                                        object_inferred_truth, source_accuracy)

        # Reverse Mapping
        self.model_output[bid] = {
            "truth": np.array([
                object_inferred_truth[i]
                for i in range(len(object_inferred_truth))
                    ], dtype=object),
            "weights": super().convert_weights(bid, source_accuracy)
        }


    def convert_input(self, index, claim):
        source_observations = {}
        for obj_claims_index, _ in enumerate(claim):
            for claim_index, _ in enumerate(claim[obj_claims_index]):
                source = index[obj_claims_index][claim_index]
                value = claim[obj_claims_index][claim_index]
                if source not in source_observations:
                    source_observations[source] = {}
                source_observations[source][obj_claims_index] = value

        return source_observations


    def _init_src_accuracy(self, source_observations,
                           object_truth, object_inferred_truth,
                           source_accuracy):
        for source_id in source_observations:
            correct = 0.0
            total = 0.0
            for oid in source_observations[source_id]:
                if oid in object_truth:
                    total += 1.0
                    object_inferred_truth[oid] = object_truth[oid]
                    if object_truth[oid] == source_observations[source_id][oid]:
                        correct += 1.0
            if total == 0.0:
                source_accuracy[source_id] = round(random.uniform(0.5, 1), 3)
            else:
                source_accuracy[source_id] = correct / total
            if source_accuracy[source_id] == 1.0:
                source_accuracy[source_id] = 0.99
            elif source_accuracy[source_id] == 0.0:
                source_accuracy[source_id] = 0.01


    def update_object_assignment(self, object_observations,
                                 object_distinct_observations,
                                 source_accuracy, object_inferred_truth):
        for oid in object_observations:
            obs_scores = {}
            for (src_id, value) in object_observations[oid]:
                if value not in obs_scores:
                    obs_scores[value] = 0.0
                if len(object_distinct_observations[oid]) == 1:
                    obs_scores[value] = 1.0
                else:
                    obs_scores[value] += math.log(
                        (len(object_distinct_observations[oid]) - 1) * source_accuracy[src_id] / (
                                    1 - source_accuracy[src_id]))

            # assign largest score
            max_value = -1
            max_index = 0
            for i, _ in obs_scores.items():
                if obs_scores[i] > max_value:
                    max_value = obs_scores[i]
                    max_index = i
            object_inferred_truth[oid] = max_index


    def update_source_accuracy(self, source_observations,
                               object_inferred_truth, source_accuracy):
        for source_id in source_observations:
            correct = 0.0
            total = 0.0
            for oid in source_observations[source_id]:
                if oid in object_inferred_truth:
                    total += 1.0
                    if object_inferred_truth[oid] == source_observations[source_id][oid]:
                        correct += 1.0
            assert total != 0.0
            source_accuracy[source_id] = correct / total
            if source_accuracy[source_id] == 1.0:
                source_accuracy[source_id] = 0.99
            elif source_accuracy[source_id] == 0.0:
                source_accuracy[source_id] = 0.01
