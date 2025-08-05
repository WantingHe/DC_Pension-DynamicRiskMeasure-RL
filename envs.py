"""
Environment


"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
import pdb  # use with set_trace() for the debugger


class TradingEnv():
    # constructor
    def __init__(self, params):
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space': np.arange(params["Ndt"]),  # time space
                       'x_space': np.linspace(0, 100, 51),  # state space -- wealth
                       'y_space': np.linspace(0, 200, 51),  # state space -- income
                       # state space -- income
                       'A_space': np.linspace(0, 100, 21),  # action space -- investment amount of risky asset
                       'I_space': np.linspace(0, 50, 21),}  # action space -- insurance policy
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    # initialization of the environment
    def reset(self, Nsims=1):
        x0 = T.ones(Nsims, device=self.device) * self.params["x0"]
        y0 = T.ones(Nsims, device=self.device) * self.params["y0"]

        return x0, y0

    def market_dynamics(self, Nsim=1, Ndt=1):
        # initialize tables for market dynamics
        S_t = T.zeros((Nsim, Ndt), dtype=T.float, requires_grad=False, device=self.device)
        Y_t = T.zeros((Nsim, Ndt), dtype=T.float, requires_grad=False, device=self.device)

        S_t[:, 0] = 1.0
        Y_t[:, 0] = 1.0
        dt = self.params["T"] / self.params["Ndt"]

        for t in range(Ndt):
            # generate BMs in market dynamics
            W_s = T.randn(Nsim, device=self.device)  # BM of the stock process
            W_i = T.randn(Nsim, device=self.device)  # BM independent from W1
            W_y = self.params["rho"] * W_s + np.sqrt(1 - self.params["rho"] ** 2) * W_i  # BM of the income process

            S_t[:, t+1] = S_t[:, t] * np.exp(self.params["mu"] * dt + self.params["sigma"] * np.sqrt(dt) * W_s).to(self.device)
            Y_t[:, t+1] = Y_t[:, t] * np.exp(self.params["mu_y"] * dt + self.params["sigma_y"] * np.sqrt(dt) * W_y.cpu()).to(self.device)

        return S_t, Y_t


    # simulation engine
    def step(self, x_t, y_t, q_xt, A_t, I_t):
        sizes = A_t.shape

        # prepare parameters for stochastic process
        dt = self.params["T"] / self.params["Ndt"]
        x_t = x_t.to(self.device)
        y_t = y_t.to(self.device)
        A_t = A_t.to(self.device)
        I_t = I_t.to(self.device)

        # set two BMs with a correlation coefficient
        # BM
        W1 = T.randn(sizes, device=self.device)  # BM of the stock process
        Wn = T.randn(sizes, device=self.device)  # BM independent from W1
        W2 = self.params["rho"] * W1 + \
             np.sqrt(1 - self.params["rho"] ** 2) * Wn  # BM of the income process in terms of W1 and Wn

        # wealth modification
        x_tp1 = (x_t + self.params["alpha"] * y_t - I_t - A_t) * np.exp(self.params["r"] * dt) + \
                A_t * np.exp(self.params["mu"] * dt + self.params["sigma"] * np.sqrt(dt) * W1.cpu()).to(self.device)

        # income modification - GBM
        y_tp1 = y_t * np.exp(self.params["mu_y"] * dt + self.params["sigma_y"] * np.sqrt(dt) * W2.cpu()).to(self.device)

        # cost - cost in wealth
        cost_t_a = x_t + self.params["alpha"] * y_t - x_tp1
        cost_t_b = - x_tp1 - np.exp(self.params["r"]) * (I_t / q_xt)

        return x_tp1, y_tp1, cost_t_a, cost_t_b

