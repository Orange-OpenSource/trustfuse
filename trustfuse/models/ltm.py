import random
import pandas as pd
from trustfuse.models.model import Model
from typing import Dict


class LTM(Model):

    def __init__(self, dataset, alpha_set=[[50, 50], [100, 10000]],
        beta_set=[10, 10], max_itr=100, burnin_count=8,
        thin_count=2, output_threshold=0.9, **kwargs):
        """Initialization

        Args:
            dataset (Dataset): _
            attributes (List[str]): list of attributes of interest
            alpha_set (list, optional): parameters that control source quality. Defaults to [[50, 50], [100, 10000]].
            beta_set (list, optional): parameters that control truth. Defaults to [10, 10].
            max_itr (int, optional): iteration of inference. Defaults to 10.
            burnin_count (int, optional): param for Collapsed Gibbs Sampling (CGS). Defaults to 8.
            thin_count (int, optional): param for CGS. Defaults to 2.
            output_threshold (float, optional): threshold to decide if a fact is True. Defaults to 0.5.
        """
        super().__init__(dataset, **kwargs)

        # Hyperparameters initialization
        self.iteration_count = max_itr
        self.thin_count = thin_count
        self.burnin_count = burnin_count
        self.threshold = output_threshold

        self.beta = {}
        for i in range(2):
            self.beta[str(i)] = beta_set[1-i]
        self.alpha = {}
        for i in range(2):
            self.alpha[str(i)] = {}
            for j in range(2):
                self.alpha[str(i)][str(j)] = alpha_set[1-i][1-j]


    def _fuse(self, dataset, bid, inputs, progress):

        self.model_output[bid] = {}

        for attr in dataset.attributes[bid]:
            truth_dict, truth_prob_dict = self.generate_truth_dict(bid,
                                                                    attr)
            source_matrix = self.generate_source_matrix(bid, attr,
                                                        truth_dict)
            results = self.perform_sampling(bid, attr, truth_dict,
                                            truth_prob_dict, source_matrix)
            infer_truth_dict, source_quality = results
            self.model_output[bid][attr] = (infer_truth_dict, source_quality)


    def get_results(self, dataset):
        unified_result = {}
        for bid, _ in self.model_output.items():
            unified_result[bid] = {
                "truth": {},
                "weights": {}
            }
            for a in self.model_output[bid]:
                unified_result[bid]["weights"] = self.model_output[bid][a][1]
                # inf_values: list
                for e, inf_values in self.model_output[bid][a][0].items():
                    if e not in unified_result[bid]["truth"]:
                        unified_result[bid]["truth"][e] = {attr: [None]
                                                           for attr in dataset.attributes[bid]}
                    unified_result[bid]["truth"][e][a] = inf_values
        return unified_result


    def generate_truth_dict(self, bid, attr):
        """Initializes truth scores

        Args:
            attr (str): name of the attribute on which reconciliation is performed
        """
        def uniform_initialization_truth():
            x = random.uniform(0, 1)
            return "1" if x >= 0.5 else "0"
        truth_dict = {}
        truth_probability_dict = {}
        attr_fact_dict, _ = self.model_input[bid]
        for key in attr_fact_dict[attr].keys():
            truth_dict[key] = uniform_initialization_truth()
            truth_probability_dict[key] = 0
        return truth_dict, truth_probability_dict


    def generate_source_matrix(self, bid, attr, truth_dict):
        """Computes confusion matrix for each source

        Args:
            attr (str): attribute on which the reconciliation is performed
        """
        # confusion matrix in the paper
        source_matrix = {}
        _, attr_claim_dict = self.model_input[bid]
        # fact
        for key1 in attr_claim_dict[attr].keys():
            # source
            for key2 in attr_claim_dict[attr][key1].keys():
                if key2 not in source_matrix:
                    source_matrix[key2] = {}
                    for i in range(2):
                        source_matrix[key2][str(i)] = {}
                        for j in range(2):
                            source_matrix[key2][str(i)][str(j)] = 0
                #truthDict[key1] = "0" or "1"
                source_matrix[key2][truth_dict[key1]][attr_claim_dict[attr][key1][key2]] += 1
        return source_matrix


    def transform_data(self, dataset, progress) -> Dict:
        """Override transform_data function of superclass Model
        for a specific transormation required for LTM."""

        for bid in progress.tqdm(dataset.data, desc="Prepare data"):

            # attribute (key): {FID: {entity: attr_value}}
            attr_fact_dict = {}
            attr_claim_dict = {}

            entity_col = dataset.entity_col_name

            # Iterate over attributes of interest
            for attr in dataset.attributes[bid]:
                # Initialization of the fusion input
                # assimilated to a primary key for facts
                # fact ID
                fid = 0
                attr_fact_dict[attr] = {}
                attr_claim_dict[attr] = {}

                fid_mapping = {}
                entity_fid = {}
                source_entity = {}

                for _, row in dataset.data[bid].iterrows():
                    if not pd.isna(row[attr]):
                        pair = (row[entity_col], row[attr])

                        if pair not in fid_mapping:
                            if row[entity_col] not in entity_fid:
                                entity_fid[row[entity_col]] = []
                            entity_fid[row[entity_col]].append(fid)
                            fid_mapping[pair] = fid
                            attr_fact_dict[attr][fid] = {row[entity_col]: row[attr]}
                            attr_claim_dict[attr][fid] = {row["Source"]: "1"}
                            fid += 1
                        else:
                            attr_claim_dict[attr][fid_mapping[pair]][row["Source"]] = "1"
                        if row["Source"] not in source_entity:
                            source_entity[row["Source"]] = []
                        source_entity[row["Source"]].append(row[entity_col])     
                for s, entities in source_entity.items():
                    for e in entities:
                        fids = entity_fid[e]
                        for f in fids:
                            if s not in attr_claim_dict[attr][f]:
                                attr_claim_dict[attr][f][s] = "0"
            # Record transformed data for fusion model input
            self.model_input[bid] = (attr_fact_dict, attr_claim_dict)


    # perform sampling
    def perform_sampling(self, bid, attr, truth_dict,
                         truth_probability_dict, source_matrix):
        """Performs Collapsed Gibbs Sampling for Truth

        Args:
            attr (str): Attribute name on which reconciliation is performed
        """

        sample_size = (self.iteration_count // self.thin_count
                       - self.burnin_count // self.thin_count)

        attr_fact_dict, attr_claim_dict = self.model_input[bid]

        for i in range(1, self.iteration_count + 1):

            for fact in attr_fact_dict[attr].keys():

                ptf = self.beta[truth_dict[fact]]
                p_tf = self.beta[str(1-int(truth_dict[fact]))]
                # source
                for key in attr_claim_dict[attr][fact]:

                    truth_val = truth_dict[fact]
                    attr_val = attr_claim_dict[attr][fact][key]

                    inv_truth_val = str(1 - int(truth_val))
                    inv_attr_val = str(1 - int(attr_val))

                    atfoc = self.alpha[truth_val][attr_val]
                    atf_oc = self.alpha[truth_val][inv_attr_val]
                    a_tfoc = self.alpha[inv_truth_val][attr_val]
                    a_tf_oc = self.alpha[inv_truth_val][inv_attr_val]

                    ntfoc = source_matrix[key][truth_val][attr_val]
                    ntf_oc = source_matrix[key][truth_val][inv_attr_val]
                    n_tfoc = source_matrix[key][inv_truth_val][attr_val]
                    n_tf_oc = source_matrix[key][inv_truth_val][inv_attr_val]

                    ptf = (ptf * (ntfoc - 1 + atfoc)
                           / (ntfoc + ntf_oc - 1 + atfoc + atf_oc))
                    p_tf = (p_tf * (n_tfoc + a_tfoc)
                            / (n_tfoc + n_tf_oc + a_tfoc + a_tf_oc))

                if random.uniform(0, 1) < p_tf / (ptf + p_tf):
                    truth_dict[fact] = str(1-int(truth_dict[fact]))

                    for key in attr_claim_dict[attr][fact]:

                        truth_val = truth_dict[fact]
                        attr_val = attr_claim_dict[attr][fact][key]
                        inv_truth_val = str(1 - int(truth_val))

                        source_matrix[key][inv_truth_val][attr_val] -= 1
                        source_matrix[key][truth_val][attr_val] += 1


                if i > self.burnin_count and i % self.thin_count == 0:
                    truth_probability_dict[fact] = (truth_probability_dict[fact]
                                                    + int(truth_dict[fact])
                                                    / sample_size)

        infer_truth_dict = self.truth_inference(bid, attr,
                                                truth_probability_dict)
        source_quality = self.source_quality_estimation(bid, attr,
                                                        truth_probability_dict)

        return infer_truth_dict, source_quality


    # infer truth
    def truth_inference(self, bid, attr, truth_probability_dict):
        attr_fact_dict, _ = self.model_input[bid]
        infer_truth_dict = {}
        for key in truth_probability_dict:
            for obj in attr_fact_dict[attr][key]:
                if obj not in infer_truth_dict:
                    infer_truth_dict[obj] = []
                if truth_probability_dict[key] >= self.threshold:
                    infer_truth_dict[obj].append(
                        attr_fact_dict[attr][key][obj])
        return infer_truth_dict


    def source_quality_estimation(self, bid, attr,
                                  truth_probability_dict):
        source_quality = {}
        source_quality_matrix = {}
        _, attr_claim_dict = self.model_input[bid]      
        for fact, _ in attr_claim_dict[attr].items():
            for source, _ in attr_claim_dict[attr][fact].items():
                if source not in source_quality_matrix:
                    source_quality[source] = {}
                    source_quality_matrix[source] = {}
                    for i in range(2):
                        source_quality_matrix[source][str(i)] = {}
                        for j in range(2):
                            source_quality_matrix[source][str(i)][str(j)] = 0

                attr_val = attr_claim_dict[attr][fact][source]
                truth_prob = truth_probability_dict[fact]

                source_quality_matrix[source]["1"][attr_val] += truth_prob
                source_quality_matrix[source]["0"][attr_val] += 1 - truth_prob

        for source, sq in source_quality.items():
            tp = source_quality_matrix[source]["1"]["1"]
            fn = source_quality_matrix[source]["1"]["0"]
            tn = source_quality_matrix[source]["0"]["0"]
            fp = source_quality_matrix[source]["0"]["1"]

            alpha_tp = self.alpha["1"]["1"]
            alpha_fn = self.alpha["1"]["0"]
            alpha_tn = self.alpha["0"]["0"]
            alpha_fp = self.alpha["0"]["1"]

            sq["recall"] = ((tp + alpha_tp)
                            / (tp + fn + alpha_tp + alpha_fn))
            sq["specificity"] = ((tn + alpha_tn)
                                 / (tn + fp + alpha_tn + alpha_fp))

        return source_quality
            