from __future__ import division
import pandas as pd
import tqdm
import numpy as np
from trustfuse.models.model import Model


class CRH(Model):

    def __init__(self, dataset, progress=tqdm, tol=1e-3,
                 max_itr=1, eps=0.1, **kwargs):
        super().__init__(dataset, progress, **kwargs)

        self.tol = tol
        self.max_itr = max_itr
        self.eps = eps

    def _fuse(self, dataset, bid, inputs, progress):

        index, claim, _ = inputs
        itr = 0
        w_vec = np.ones(self.source_nb[bid]) / self.source_nb[bid]
        # Initialization of truth
        truth = np.zeros(self.fact_nb[bid], dtype=object)
        self.initialize_truth(truth, claim, bid)
        while itr < self.max_itr:
            itr = itr+1
            #truth_old = np.copy(truth)
            # 1st step
            w_vec = self.update_w(claim, index, truth, bid)
            # 2nd step
            truth = self.update_truth(claim, index, w_vec, bid)
            #err = la.norm(truth-truth_old)/la.norm(truth_old)
        self.model_output[bid] = {
            "truth": truth,
            "weights": w_vec
        }


    def update_w(self, claim, index, truth, bid):
        # rtn = future w_vec updated
        rtn = np.zeros(self.source_nb[bid])
        for i in range(self.fact_nb[bid]):
            # Select the right dictance function in rekation to the data type
            # If categorical data
            if isinstance(claim[i][0], str):
                # Eq(12) 0-1 loss strategy
                loss_01 = np.array(claim[i] != truth[i], dtype=int)
                rtn[index[i]] = rtn[index[i]] + loss_01
            elif isinstance(claim[i][0], (float, int)):
                # (claim[i]-truth[i])**2/max(np.std(claim[i]),eps) = eq (19) of CRH paper
                # truth[i] = vim(*)
                rtn[index[i]] = rtn[index[i]] \
                    + ((claim[i] - truth[i]) ** 2) \
                    / max(np.std(claim[i]), self.eps)

        # tmp = Σ(k)Σ(n)[dm(vim(*), v)]
        # Generate data
        tmp = np.sum(rtn)
        if tmp > 0:
            rtn[rtn>0] = np.copy(-np.log(rtn[rtn>0] / tmp))
        return rtn


    def update_truth(self, claim, index, w_vec, bid):
        rtn = [None for _ in range(self.fact_nb[bid])]
        for i in range(self.fact_nb[bid]):
            # If categorical data including string
            if isinstance(claim[i][0], str):
                all_possible_values = list(set(claim[i]))
                possible_truth_indexes = np.zeros(len(all_possible_values))
                for ind, val in enumerate(all_possible_values):
                    loss_01 = np.array(claim[i] == val, dtype=int)
                    possible_truth_indexes[ind] = np.sum(np.multiply(w_vec[index[i]], loss_01))
                truth_index = np.argmax(possible_truth_indexes)
                rtn[i] = all_possible_values[truth_index]
            # If numerical data
            elif isinstance(claim[i][0], (float, int)):
                # Eq(3)
                all_possible_values = list(claim[i])
                possible_truth_indexes = np.zeros(len(all_possible_values))
                for ind, val in enumerate(all_possible_values):
                    loss = (val - claim[i]) ** 2 / max(np.std(claim[i]), self.eps)
                    possible_truth_indexes[ind] = np.sum(np.multiply(w_vec[index[i]], loss))
                truth_index = np.argmin(possible_truth_indexes)
                rtn[i] = all_possible_values[truth_index]
                # Eq(20)
                #rtn[i] = np.dot(w_vec[index[i]],claim[i])/np.sum(w_vec[index[i]])
        return rtn


    def initialize_truth(self, truth, claim, bid):
        for i in range(self.fact_nb[bid]):
            # Majority voting
            if isinstance(claim[i][0], str):
                strings_series = pd.Series(claim[i])
                most_frequent = strings_series.mode()
                truth[i] = most_frequent.iloc[0]
            elif isinstance(claim[i][0], (float, int)):
                # Median
                truth[i] = np.median(claim[i])
                # Mean
                #truth[i] = np.mean(claim[i])
