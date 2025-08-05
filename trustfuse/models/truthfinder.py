from __future__ import division

import numpy as np
from trustfuse.models.model import Model
from trustfuse.models.utils import jaro_distance


class TruthFinder(Model):

    def __init__(self, dataset, tol=0.1, max_itr=10, rho=0.5,
                 gamma=0.3, base_threshold=0, **kwargs):

        super().__init__(dataset, **kwargs)

        self.max_itr = max_itr
        self.tol = tol
        self.rho = rho
        self.gamma = gamma
        self.base_threshold = base_threshold


    def _fuse(self, dataset, bid, inputs, progress):

        index, claim, _ = inputs
        itr = 0
        tau_vec = - np.log(1 - np.ones(self.source_nb[bid]) * 0.9)
        while itr < self.max_itr:
            itr += 1
            # tau_old = np.copy(tau_vec)
            s_set = self.update_claim(claim, index, tau_vec, bid)
            tau_vec = self.update_source(index, s_set, bid)
            # err = 1 - np.dot(tau_vec, tau_old) / (la.norm(tau_vec) * la.norm(tau_old))
        truth = np.zeros(self.fact_nb[bid], dtype=object)
        for i in range(self.fact_nb[bid]):
            truth[i] = claim[i][np.argmax(s_set[i])]

        self.model_output[bid] = {
            "truth": truth,
            "weights": tau_vec
        }


    def update_source(self, index, s_set, bid):
        t_vec = np.zeros(self.source_nb[bid])
        tau_vec = np.zeros(self.source_nb[bid])
        count = np.zeros(self.source_nb[bid])
        for i in range(self.fact_nb[bid]):
            t_vec[index[i]] = t_vec[index[i]] + s_set[i]
            count[index[i]] = count[index[i]] + 1
        t_vec[count > 0] = t_vec[count > 0] / count[count > 0]
        tau_vec[t_vec >= 1] = np.log(1e10)
        tau_vec[t_vec < 1] = - np.log(1 - t_vec[t_vec < 1])

        return tau_vec


    def update_claim(self, claim, index, tau_vec, bid):
        s_set= []
        for i in range(self.fact_nb[bid]):
            # Get distinct values (fact reduction)
            claim_set = list(set(claim[i]))
            # Create sigma for each distinct fact
            sigma_i = np.zeros(len(claim_set))
            # Same thing s = 1 - e^(-sigma) eq. (7)
            s_vec = np.zeros(len(claim[i]))
            # Iterates over the facts of the (entity, attribute)
            for j, _ in enumerate(claim_set):
                # Equation (1)
                # source selection that provide the fact f(j)
                sigma_i[j] = sum(tau_vec[index[i]][claim[i] == claim_set[j]])
            tmp_i = np.copy(sigma_i)
            for j, _ in enumerate(claim_set):
                if self.claim_type[bid][i] in ["string", "entity"]:
                    loss_sum = 0
                    for k, _ in enumerate(claim_set):
                        if k != j:
                            loss_sum += (sigma_i[k] *
                                         (np.exp(-jaro_distance(claim_set[j], claim_set[k]))
                                          - self.base_threshold))

                    tmp_i[j] = ((1 - self.rho
                                 * (1 - self.base_threshold))
                                 * sigma_i[j]
                                 + self.rho
                                 * loss_sum)

                elif self.claim_type[bid][i] == "quantity":

                    tmp_i[j] = ((1 - self.rho * (1 - self.base_threshold))
                                * sigma_i[j]
                                + self.rho
                                * sum((np.exp(-abs(claim_set - claim_set[j]))
                                       - self.base_threshold)
                                       * sigma_i))

                s_vec[claim[i]==claim_set[j]] = 1 / (1 + np.exp(-self.gamma * tmp_i[j]))

            s_set.append(s_vec)

        return s_set
