from __future__ import division
import numpy as np
from trustfuse.models.model import Model


class KDEm(Model):

    def __init__(self, dataset, tol=1e-5, max_itr=99, method="Gaussian",
                 h=-1, **kwargs):

        super().__init__(dataset, **kwargs)

        self.tol = tol
        self.max_itr = max_itr
        self.method = method
        self.h = h


    def _fuse(self, dataset, bid, inputs, progress):

        err = 99
        index, claim, count = inputs
        w_m = []
        for i in range(self.fact_nb[bid]):
            l = len(index[i])
            # source weights vector
            w_m.append(np.ones(l) / l)
        itr = 1
        kernel_m = self.get_kernel_matrix(claim, bid)
        norm_m = self.get_norm_matrix(kernel_m, bid, w_m)
        c_vec, j = self.update_c(index, count, bid, norm_m)
        while err > self.tol and itr < self.max_itr:
            itr += 1
            j_old = j
            c_old = np.copy(c_vec)
            w_m = self.update_w(index, bid, c_old, norm_m)
            norm_m = self.get_norm_matrix(kernel_m, bid, w_m)
            c_vec, j = self.update_c(index, count, bid, norm_m)
            # err = la.norm(c_vec - c_old) / la.norm(c_old)
            err = abs((j - j_old) / j_old)

        self.model_output[bid] = {
            "truth": c_vec,
            "weights": w_m
        }


    def get_kernel_matrix(self, claim, bid):
        kernel_m = []
        for i in range(self.fact_nb[bid]):
            x_i = claim[i]
            if self.h < 0:
                self.h = self.mad(x_i)
                # self.h = np.std(x_i)
            l = x_i.shape[0]
            tmp = np.zeros((l, l))
            for j in range(l):
                if self.h > 0:
                    tmp[j, :] = self.k((x_i[j] - x_i) / self.h)
                else:
                    tmp[j, :] = self.k(0)
            kernel_m.append(tmp)
        return kernel_m


    def get_norm_matrix(self, kernel_m, bid, w_m):
        norm_m = []
        for i in range(self.fact_nb[bid]):
            kernel = kernel_m[i]
            term1 = np.diag(kernel)
            term2 = np.dot(kernel, w_m[i])
            term3 = np.dot(w_m[i], term2)
            tmp = term1 - 2 * term2 + term3
            tmp[tmp < 0] = 0
            norm_m.append(tmp)
        return norm_m


    def k(self, x):
        rtn = 0
        if self.method.lower() == "uniform":
            rtn = (abs(x) <= 1) / 2
        if self.method.lower() == "epanechnikov" or self.method.lower() == "ep":
            rtn = 3 / 4 * (1 - x ** 2) * (abs(x) <= 1)
        if self.method.lower() == "biweight" or self.method.lower() == "bi":
            rtn = 15 / 16 * (1 - x ** 2) ** 2 * (abs(x) <= 1)
        if self.method.lower() == "triweight" or self.method.lower() == "tri":
            rtn = 35 / 32 * (1 - x ** 2) ** 3 * (abs(x) <= 1)
        if self.method.lower() == "gaussian":
            rtn = np.exp(-x ** 2) / np.sqrt(2 * np.pi)
        if self.method.lower() == "laplace":
            rtn = np.exp(-abs(x))
        return rtn


    def mad(self, x_i):
        return (np.median(abs(x_i - np.median(x_i, axis=0)), axis=0)
                + 1e-10
                * np.std(x_i, axis=0))


    # update source reliability scores
    def update_c(self, index, count, bid, norm_m):
        rtn = np.zeros(self.source_nb[bid])
        for i in range(self.fact_nb[bid]):
            rtn[index[i]] = rtn[index[i]] + norm_m[i] / len(index[i])
        tmp = np.sum(rtn)
        if tmp > 0:
            rtn[rtn > 0] = np.copy(- np.log((rtn[rtn > 0]
                                            / count[rtn > 0])
                                            / tmp))
        return [rtn, tmp]

    # update opinion distributions
    def update_w(self, index, bid, c_vec, norm_m):
        w_m = []
        for i in range(self.fact_nb[bid]):
            w_i = np.zeros(len(index[i]))
            tmp = c_vec[index[i]]
            w_i[norm_m[i] > 0] = tmp[norm_m[i] > 0]
            tmp1 = sum(w_i)
            if tmp1 > 0 :
                w_m.append(w_i / tmp1)
            else:
                w_i[norm_m[i] == 0] = 1
                tmp1 = sum(w_i)
                w_m.append(w_i / tmp1)
        return w_m
