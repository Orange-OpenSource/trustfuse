import random
import math
import numpy as np
from trustfuse.models.model import Model


class SLiMFast(Model):

    def __init__(self, dataset, source_features=None,
                 alpha=0.01, reg=0.01, iterations=10, **kwargs):

        super().__init__(dataset, **kwargs)
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations


    def _fuse(self, dataset, bid, inputs, progress):

        # claim[i] = one object
        index, claim, _ = inputs
        source_observations, source_features = self.convert_input(index,
                                                                    claim)
        # load data
        object_truth = {}
        # init local dictionaries
        object_inferred_truth = {}
        object_distinct_observations = {}
        object_observations = {}
        # object to value dictionary
        for sid, _ in source_observations.items():
            for oid in source_observations[sid]:
                if oid in object_truth:
                    continue
                if oid not in object_observations:
                    object_observations[oid] = []
                    object_distinct_observations[oid] = set([])
                    object_inferred_truth[oid] = \
                    source_observations[sid][oid]
                object_observations[oid].append(
                    (sid, source_observations[sid][oid]))
                object_distinct_observations[oid].add(
                    source_observations[sid][oid])

        # initialize source accuracy, utilize any ground truth if specified
        source_accuracy, feature_weights = self._init_feature_weights(source_features)

        # Solve
        if len(object_truth) > 0:
            # If GT is provided
            self.update_source_accuracy(object_truth,
                                        source_features,
                                        feature_weights,
                                        object_observations,
                                        source_observations,
                                        source_accuracy)

        for _ in range(self.iterations):
            self.update_object_assignment(object_observations,
                                            object_distinct_observations,
                                            source_accuracy,
                                            object_inferred_truth)

            self.update_source_accuracy(object_inferred_truth,
                                        source_features,
                                        feature_weights,
                                        object_observations,
                                        source_observations,
                                        source_accuracy)

        self.model_output[bid] = {
            "truth": np.array([
                object_inferred_truth[i]
                for i in range(len(object_inferred_truth))
                ], dtype=object),
            "weights": super().convert_weights(bid, source_accuracy)
        }


    def convert_input(self, index, claim):
        source_observations = {}
        source_features = {}
        for obj_claims_index, _ in enumerate(claim):
            for claim_index, _ in enumerate(claim[obj_claims_index]):
                source = index[obj_claims_index][claim_index]
                value = claim[obj_claims_index][claim_index]
                if source not in source_observations:
                    source_observations[source] = {}
                source_observations[source][obj_claims_index] = value
                if source not in source_features:
                    source_features[source] = {"w": 0.}

        return source_observations, source_features


    def _init_feature_weights(self, source_features):
        source_accuracy = {}
        feature_weights = {}
        for sid in source_features:
            for feat in source_features[sid]:
                feature_weights[feat] = 0.0
            source_accuracy[sid] = round(random.uniform(0.7, 0.99), 3)
        return source_accuracy, feature_weights


    # *** Equation (3) ***
    def update_feature_weights(self, sid, correct,
                               source_features, feature_weights):
        total_weight = 0.0
        for feat in source_features[sid]:
            total_weight += feature_weights[feat]
        for feat in source_features[sid]:
            if correct:
                feature_weights[feat] -= self.alpha * (
                        -1.0 / (math.exp(-1.0 * total_weight) + 1.0))
            else:
                feature_weights[feat] -= self.alpha * (
                        1.0 / (1.0 + math.exp(-1.0 * total_weight)))


    # *** Equation (2) of the paper ***
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
                        ((len(object_distinct_observations[oid]) - 1) * \
                         source_accuracy[src_id]
                         / (1 - source_accuracy[src_id]))
                        )

            # assign largest score
            object_inferred_truth[oid] = max(obs_scores, key=obs_scores.get)


    def update_source_accuracy(self, object_truth,
                               source_features, feature_weights,
                               object_observations, source_observations,
                               source_accuracy):
        for oid in object_observations:
            if oid in object_truth:
                for sid, value in object_observations[oid]:
                    # Need to know GT
                    if object_truth[oid] == value:
                        self.update_feature_weights(sid, True,
                                                    source_features,
                                                    feature_weights)
                    else:
                        self.update_feature_weights(sid, False,
                                                    source_features,
                                                    feature_weights)
        for feat in feature_weights:
            # L1-regularization
            if feature_weights[feat] > 0:
                feature_weights[feat] = max(0, feature_weights[
                    feat] - self.alpha * self.reg)
            elif feature_weights[feat] < 0:
                feature_weights[feat] = min(0, feature_weights[
                    feat] + self.alpha * self.reg)
        for sid in source_observations:
            total_weight = 0.0
            for feat in source_features[sid]:
                total_weight += feature_weights[feat]
            source_accuracy[sid] = 1.0 / (
                    1.0 + math.exp(-total_weight))
            if source_accuracy[sid] == 1.0:
                source_accuracy[sid] = 0.99
            elif source_accuracy[sid] == 0.0:
                source_accuracy[sid] = 0.01


def print_source_accuracy(method):
    for source in method.source_accuracy:
        print(source + ","+ str(method.source_accuracy[source]))
