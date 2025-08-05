from __future__ import division
import numpy as np
import numpy.linalg as la
from scipy.stats import chi2
from trustfuse.models.model import Model


class CATD(Model):

    def __init__(self, dataset, tol=0.1, max_itr=10, eps=1e-15, **kwargs):
        super().__init__(dataset, **kwargs)

        self.tol = tol
        self.max_itr = max_itr
        self.eps = eps


    def _fuse(self, dataset, bid, inputs, progress):

        index, claim, count = inputs

        w_vec = np.ones(self.source_nb[bid])
        truth = self.initialize_truth(claim, bid)
        err = 99
        itr = 0
        while err > self.tol and itr < self.max_itr:
            w_old = np.copy(w_vec)
            w_vec = self.update_w(claim, index, count, truth, bid)
            truth = self.update_truth(claim, index, w_vec, bid)
            err = la.norm(w_old - w_vec) / la.norm(w_old)
            itr = itr+1
        truth = np.array([claim[i][np.abs(claim[i] - truth[i]).argmin()]
                            for i in range(len(truth))])
        self.model_output[bid] = {
            "truth": truth,
            "weights": w_vec
        }


    def update_w(self, claim, index, count, truth, bid):
        rtn = np.zeros(self.source_nb[bid])
        for i in range(self.fact_nb[bid]):
            rtn[index[i]] = rtn[index[i]] + (claim[i] - truth[i]) **2
        # where alpha = 0.05
        rtn[rtn>0] = chi2.cdf(0.025, count[rtn>0]) / rtn[rtn>0]
        rtn[rtn==0] = 1e10
        return rtn


    def update_truth(self, claim, index, w_vec, bid):
        rtn = np.zeros(self.fact_nb[bid])
        for i in range(self.fact_nb[bid]):
            rtn[i] = (np.dot(w_vec[index[i]], claim[i])
                      / np.sum(w_vec[index[i]]))
        return rtn


    def initialize_truth(self, claim, bid):
        truth = np.zeros(self.fact_nb[bid])
        for i in range(self.fact_nb[bid]):
            truth[i] = np.median(claim[i])
        return truth
