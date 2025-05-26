"""
Risk measures
Implementation of the mean, CVaR, penalized-CVaR, semi-dev and mean-CVaR<

"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
# import cvxpy as cp
from scipy import optimize
import pdb  # use with set_trace() for the debugger


class RiskMeasure():
    # constructor
    def __init__(self, params, Type_a, Type_b, alpha=[0.2, 0.2], kappa=[1, 1], r=[2, 2]):
        self.Type_a = Type_a
        self.Type_b = Type_b
        self.Type = [self.Type_a, self.Type_b]
        self.alpha = alpha
        self.kappa = kappa
        self.r = r
        self.params = params

        for idx in range(len(self.Type)):
            # Conditional value-at-risk
            if (self.Type[idx] == 'CVaR'):
                assert (alpha[idx] > 0) and (alpha[idx] < 1), "alpha needs to be in (0,1)"
                self.alpha[idx] = alpha[idx]

            else:
                assert False, "Type of the risk measure is unknown"


    # calculate the risk of a sequence of values
    def compute_risk_a(self, x):

        # Conditional value-at-risk
        if (self.Type_a == 'CVaR'):
                quant = T.quantile(x, 1 - self.alpha[0], axis=1).unsqueeze(1).repeat(1, x.shape[1])
                cond = x >= quant
                RM = T.sum(x.masked_fill(~cond, 0.0), axis=1) / T.sum(cond, axis=1)

        return RM


    def compute_risk_b(self, x):
        # Conditional value-at-risk
        if (self.Type_b == 'CVaR'):
            quant = T.quantile(x, 1 - self.alpha[1], axis=1).unsqueeze(-1).repeat(1, x.shape[1])
            cond = x >= quant
            RM = T.sum(x.masked_fill(~cond, 0.0), axis=1) / T.sum(cond, axis=1)

        return RM


    # calculate the gradient based on transitions and rewards
    def get_V_loss(self, V_tp1, cost_t_b, logprob, q_xt):
        # Conditional value-at-risk
        if(self.Type == ['CVaR','CVaR']):
            quant_A = T.quantile(V_tp1, 1 - self.alpha[0], axis=1).unsqueeze(1).repeat(1, V_tp1.shape[1], 1)
            quant_B = T.quantile(cost_t_b, 1 - self.alpha[1], axis=1).unsqueeze(1).repeat(1, cost_t_b.shape[1], 1)

            cond_A = V_tp1 > quant_A
            cond_B = cost_t_b > quant_B

            Z_A = V_tp1.clone().masked_fill(~cond_A, 0.0)
            Z_B = cost_t_b.clone().masked_fill(~cond_B, 0.0)

            loss_A = T.sum(logprob.masked_fill(~cond_A, 0.0) * (1 - q_xt) * (Z_A - quant_A), axis=1) / (self.alpha[0] * V_tp1.shape[1])  # T.sum(cond_A, axis=1)
            loss_B = T.sum(logprob.masked_fill(~cond_B, 0.0) * q_xt * (Z_B - quant_B), axis=1) / (self.alpha[1] * cost_t_b.shape[1])  # T.sum(cond_B, axis=1)
            loss = loss_A + loss_B

        else:
            pass

        return loss

