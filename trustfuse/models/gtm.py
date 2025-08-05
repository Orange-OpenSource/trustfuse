from __future__ import division
import numpy as np
import numpy.linalg as la
from trustfuse.models.model import Model


class GTM(Model):

    def __init__(self, dataset, tol=1e-3, max_itr=99, alpha=10,
                 beta=10, mu0=0, sigma0=1, **kwargs):

        super().__init__(dataset, **kwargs)

        self.tol = tol
        self.max_itr = max_itr
        self.alpha = alpha
        self.beta = beta
        self.mu0 = mu0
        self.sigma0 = sigma0


    def _fuse(self, dataset, bid, inputs, progress):

        index, claim, _ = inputs
        initial_claim = claim.copy()
        itr = 0
        truth, claim, index, prior_truth, sigma_e = self.data_preprocessing(claim, index, bid)
        truth, sigma_vec = self.initialization(truth, claim, index, bid)
        err = 99

        while err > self.tol and itr < self.max_itr:
            itr += 1
            truth_old = np.copy(truth)
            truth = self.expectation(claim, index, sigma_vec, bid)
            sigma_vec = self.maximization(claim, index, truth, bid)
            err = la.norm(truth - truth_old) / la.norm(truth_old)

        # Post-processing to rescale the truth
        truth = truth * sigma_e + prior_truth

        truth = np.array([initial_claim[i][np.abs(initial_claim[i] - truth[i]).argmin()]
                            for i in range(len(truth))])

        self.model_output[bid] = {
            "truth": truth,
            "weights": sigma_vec
        }



    def expectation(self, claim, index, sigma_vec, bid):
        truth = np.zeros(self.fact_nb[bid])
        for i in range(self.fact_nb[bid]):
            tmp = (self.mu0 / self.sigma0 ** 2
                   + sum(claim[i] / sigma_vec[index[i]] ** 2))
            tmp1 = 1 / self.sigma0 ** 2 + sum(1 / sigma_vec[index[i]] ** 2)
            truth[i] = tmp / tmp1
        return truth


    def maximization(self, claim, index, truth, bid):
        sigma_vec = np.zeros(self.source_nb[bid])
        count = np.zeros(self.source_nb[bid])
        for i in range(self.fact_nb[bid]):
            sigma_vec[index[i]] = (sigma_vec[index[i]]
                                   + 2 * self.beta
                                   + (claim[i] - truth[i]) ** 2)
            count[index[i]] = count[index[i]] + 1
        sigma_vec = sigma_vec / (2 * (self.alpha + 1) + count)
        return sigma_vec


    def initialization(self, truth, claim, index, bid):
        sigma_vec = self.maximization(claim, index, truth, bid)
        return truth, sigma_vec


    def data_preprocessing(self, claim, index, bid, 
                           delta0=1, delta2=2, strategy="median"):
        """ 
        Corresponds to Algorithm 1 Data processing of the paper:
        A Probabilistic Model for Estimating Real-valued Truth from Conflicting Sources
        """

        # Initialize the Gaussian parameters
        prior_truth = np.zeros(self.fact_nb[bid])
        sigma_e = np.zeros(self.fact_nb[bid])
        if strategy == "median":
            for i in range(self.fact_nb[bid]):
                prior_truth[i] = np.median(claim[i])

        for i in range(self.fact_nb[bid]):
            # To register outliers for the n entities that will be used
            # as a mask to select data without outliers, False = outlier (mask)
            outliers = np.ones(len(claim[i]), dtype=bool)
            for c, vc in enumerate(claim[i]):
                if prior_truth[i] != 0 and abs(vc - prior_truth[i]) / prior_truth[i] > delta0:
                    outliers[c] = False

            # Update claims by removing outliers
            claim[i] = claim[i][outliers]
            index[i] = np.array(index[i])[outliers]
            sigma_e[i] = np.std(claim[i])
            while np.all(outliers) is False:
                # Initialize outliers
                outliers = np.ones(len(claim[i]), dtype=bool)
                for c, vc  in enumerate(claim[i]):
                    if sigma_e[i] > 0 and abs(vc - prior_truth[i]) / sigma_e[i] > delta2:
                        outliers[c] = False

                claim[i] = claim[i][outliers]
                index[i] = index[i][outliers]
                sigma_e[i] = np.std(claim[i])
            index[i] = list(index[i])
        # Normalization
        claim_normalized = []
        for i in range(self.fact_nb[bid]):
            claim_n = np.zeros(len(claim[i]))
            for c, vc  in enumerate(claim[i]):
                claim_n[c] = vc - prior_truth[i] #/ sigma_e[i]
            claim_normalized.append(claim_n)

        return prior_truth, claim_normalized, index, prior_truth, sigma_e
