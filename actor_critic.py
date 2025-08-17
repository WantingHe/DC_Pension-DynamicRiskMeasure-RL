"""
# Policy gradient functions (actor-critic style algorithm)


"""
import os
# numpy
import numpy as np
import numpy.matlib
# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# pytorch
import torch as T
import torch.optim as optim
import random
from torch.distributions import MultivariateNormal
# misc
import utils
from datetime import datetime
import pdb  # use with set_trace() for the debugger


class ActorCriticPG():
    # constructor
    def __init__(self,
                 params,
                 mortality,  # list of mortality rates
                 repo,  # repository for files
                 method,  # sub folder for files
                 env,  # environment
                 policy,  # ANN structure for the policy
                 V,  # ANN structure for the value function
                 risk_measure,  # risk measure
                 gamma=1,  # discount factor
                 rng_seed=None):  # replication purpose

        assert (gamma > 0) and (gamma <= 1), "gamma needs to be in (0,1]"

        # assign objects to the actor_critic instance
        self.params = params
        self.mortality = mortality
        self.policy = policy  # policy (ACTOR)
        self.V = V  # value function (CRITIC)
        self.env = env  # environment
        self.repo = repo  # repository for files
        self.method = method  # sub folder for files
        self.risk_measure = risk_measure  # risk measure
        self.gamma = gamma  # discount factor
        self.device = self.policy.device  # PyTorch device
        self.seed = rng_seed  # replication purpose

        # initialize loss objects
        self.loss_history_policy = []  # keep track of all losses for the policy
        self.loss_history_V = []  # keep track of all losses for the V
        self.loss_trail = 100  # number of epochs for the loss moving average
        self.loss_print = 50  # number of epochs before printing the loss
        self.gradients = []  # keep track of all losses gradients for the policy

        # create lists and dictionaries for optimal policy
        self.states_list = []
        self.V_opt = {}
        self.best_actions = {}


    # select an action according to the policy ('best' or 'random')
    def select_actions(self, x_t, y_t, time_t, choose, seed=None):
        assert x_t.shape[0] == y_t.shape[0], "x and y must have same shape"
        assert y_t.shape[0] == time_t.shape[0], "y and time must have same shape"

        # freeze the set of random normal variables
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        # observations as a formatted tensor
        obx_t = T.stack((x_t.to(self.device).clone(), y_t.to(self.device).clone(), time_t.to(self.device).clone()), -1)

        # obtain parameters of the bivariate normal distribution
        A_mean, A_std, I_mean, I_std, correlation = self.policy(obx_t.clone(), x_t.clone() + self.params["alpha"] * y_t.clone())

        # create action distributions with a bivariate Normal distribution
        mean = T.stack((A_mean.clone(), I_mean.clone()), -1)
        covar = T.stack((T.stack((T.square(A_std.clone()), correlation * A_std.clone() * I_std.clone()), -1),
                          T.stack((correlation.clone() * A_std.clone() * I_std.clone(), T.square(I_std.clone())), -1)), -1)

        jitter = 1e-6
        covar = covar + T.eye(covar.size(-1)).to(covar.device) * jitter

        action_dist = MultivariateNormal(mean, covar)

        # get action from the policy
        if choose == 'random':
            if seed is not None:
                T.manual_seed(seed)
                np.random.seed(seed)
            action_sample = action_dist.rsample() # .view(covar.shape[0], covar.shape[1], -1)  # random sample from the Normal and MultiNormal
        elif choose == 'best':
            action_sample = mean  # mode of the Normal amd MultiNormal
        else:
            assert False, "Type of action selection is unknown ('random' or 'best')"

        # get actions from the mapping: (-infty, infty) -> (min, max)
        lower_bound_I = T.zeros(action_sample[:, :, 1].squeeze(-1).shape, device=self.device)
        upper_bound_I = (x_t.squeeze(-1) + self.env.params["alpha"] * y_t.squeeze(-1)).to(self.device)
        I_t = T.max(lower_bound_I, T.min(action_sample[:, :, 1].squeeze(-1), upper_bound_I))

        lower_bound_A = T.zeros(action_sample[:, :, 0].squeeze(-1).shape, device=self.device)
        upper_bound_A = upper_bound_I - I_t
        A_t = T.max(lower_bound_A, T.min(action_sample[:, :, 0].squeeze(-1), upper_bound_A))

        # get log-probabilities of the action
        log_prob_t = action_dist.log_prob(action_sample.detach()).squeeze()

        # verification of any problem with log_prob
        if (T.isnan(log_prob_t).any() or T.isinf(log_prob_t).any()):
            assert False, "missing or infinite values in the gradients"

        return A_t, I_t, log_prob_t


############################################ Step 1.2 ################################################
    # simulate trajectories from the policy
    def sim_trajectories(self,
                         Ntrajectories=100,  # number of trajectories
                         Mtransitions=100,  # number of transitions
                         choose='random',  # how to choose the actions
                         seed=None):  # random seed

        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        # initialize tables for all trajectories
        x = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                    dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        y = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                    dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        q_xt = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                    dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        timestep = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                           dtype=T.float, requires_grad=False, device=self.device).to(self.device)

        x_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                        dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        y_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                        dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        timestep_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                               dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        A_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                      dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        I_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                       dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        log_prob_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                             dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        cost_t_a = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                         dtype=T.float, requires_grad=False, device=self.device).to(self.device)
        cost_t_b = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                           dtype=T.float, requires_grad=False, device=self.device).to(self.device)

        # simulate N trajectories
        x_sim, y_sim = self.env.reset(Ntrajectories)
        q_xt_sim = self.mortality

        # if Ntrajectories < q_xt_sim.shape[0]:
        batch_idx = np.random.choice(q_xt_sim.shape[0], size=Ntrajectories, replace=False)
        q_xt_sim = q_xt_sim[batch_idx, :]


        for t_idx in self.env.spaces["t_space"]:
            # starting state (outer)
            x[:, t_idx] = x_sim.to(self.device)
            y[:, t_idx] = y_sim.to(self.device)
            q_xt[:, t_idx] = T.tensor(q_xt_sim[:, t_idx], device=self.device)
            timestep[:, t_idx] = t_idx

            # get actions from the policy (outer)
            A_traj, I_traj, _ = self.select_actions(x_sim.unsqueeze(-1), y_sim.unsqueeze(-1),
                                                     t_idx * T.ones(Ntrajectories, device=self.device).unsqueeze(-1),
                                                     'random')

            # get state variables for the next time step from the dynamics (outer)
            x_sim, y_sim, _, _ = self.env.step(x_sim, y_sim, q_xt[:, t_idx], A_traj, I_traj)
            timestep_tp1[:, :, t_idx] = t_idx + 1

        for t_idx in self.env.spaces["t_space"]:
            # get actions from the policy (inner)
            A_t[:, :, t_idx], I_t[:, :, t_idx], log_prob_t[:, :, t_idx] = \
                self.select_actions(x[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                                    y[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                                    timestep[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                                    choose)

            # simulate transitions (inner): multiple actions
            x_tp1[:, :, t_idx], y_tp1[:, :, t_idx], cost_t_a[:, :, t_idx], cost_t_b[:, :, t_idx] = \
                self.env.step(x[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                              y[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                              q_xt[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                              A_t[:, :, t_idx],
                              I_t[:, :, t_idx])

            timestep_tp1[:, :, t_idx] = t_idx + 1

        # store (outer) trajectories in a dictionary
        trajs = {'x': x,  # starting and ending states -- wealth
                 'y': y,  # starting and ending states -- income
                 'mortality': q_xt,  # stochastic mortality list
                 'timestep': timestep}  # starting and ending states -- time index

        # store (inner) transitions in a dictionary
        transitions = {'x_tp1': x_tp1,  # ending states from the actions -- wealth
                       'y_tp1': x_tp1,  # ending states from the actions -- income
                       'timestep_tp1': timestep_tp1,  # ending states from the actions -- time index
                       'cost_t_a': cost_t_a,  # costs from the actions when the policyholder is alive at time t+1
                       'cost_t_b': cost_t_b,  # costs from the actions when the policyholder is dead at time t+1
                       'A_t': A_t,  # actions taken in proportion
                       'I_t': I_t,  # actions taken in insurance
                       'log_prob_t': log_prob_t}  # log-prob from the actions

        return trajs, transitions

    # estimate the value function for all time steps (critic)
    def estimate_V(self,
                   Ntrajectories,  # number of trajectories
                   Mtransitions,  # number of transitions
                   batch_size=50,  # batch size for the update
                   Nepochs=100,  # number of epochs
                   rng_seed=None):  # random seed
        # print progress
        print('--Estimation of V--')
        batch_size = np.minimum(batch_size, Ntrajectories)

        # set V in training mode
        self.V.train()

        # generate full trajectories from policy
        trajs, transitions = self.sim_trajectories(Ntrajectories,
                                                   Mtransitions,
                                                   choose="random",
                                                   seed=rng_seed)

        # Define the exponential learning rate scheduler
        gamma = 0.9
        step_size = 10
        scheduler = optim.lr_scheduler.StepLR(optimizer=self.V.optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(Nepochs):
            # zero grad
            self.V.zero_grad()

            # sample a batch of states at time t+1
            batch_idx = np.random.choice(Ntrajectories, size=batch_size, replace=False)
            x_batch = trajs["x"][batch_idx, :]
            y_batch = trajs["y"][batch_idx, :]
            time_batch = trajs["timestep"][batch_idx, :]
            mortality_batch = trajs["mortality"][batch_idx, :]

            # compute predicted values
            obx_t = T.stack((x_batch, y_batch, time_batch), -1).detach()
            v_pred = self.V(obx_t).squeeze()

            # compute target values
            v_target = T.zeros(v_pred.shape, requires_grad=False)

            # value function at the next time step
            obx_tp1 = T.stack((transitions["x_tp1"][batch_idx, :, :-1],
                               transitions["y_tp1"][batch_idx, :, :-1],
                               transitions["timestep_tp1"][batch_idx, :, :-1]), -1)
            v_tp1 = self.V(obx_tp1.clone()).squeeze()
            cost_t_a = transitions["cost_t_a"][batch_idx, :, :-1]
            cost_t_b = transitions["cost_t_b"][batch_idx, :, :-1]

            # value function for other time steps
            for col_idx in range(v_target.shape[1] - 1):
                v_target[:, col_idx] = mortality_batch[:, col_idx] * self.risk_measure.compute_risk_b(cost_t_b[:, :, col_idx]) + \
                     (1 - mortality_batch[:, col_idx]) * self.risk_measure.compute_risk_a(cost_t_a[:, :, col_idx] + np.exp(-self.params["r"])*v_tp1[:, :, col_idx])

            # value function for the last time step
            v_target[:, -1] = mortality_batch[:, -1] * self.risk_measure.compute_risk_b(transitions["cost_t_b"][batch_idx, :, -1]) + \
                (1 - mortality_batch[:, -1]) * self.risk_measure.compute_risk_a(
                transitions["cost_t_a"][batch_idx, :, -1] + \
                np.exp(-self.params["r"])*self.risk_measure.compute_risk_a(-transitions["x_tp1"][batch_idx, :, -1]))

            # calculate the loss function
            v_loss = self.V.loss(v_target.detach().to(self.device), v_pred.to(self.device))
            v_loss.backward()
            self.V.optimizer.step()
            self.loss_history_V.append(v_loss.detach().cpu().numpy())

            # adjust the learning rate
            scheduler.step()

            # print progress
            if epoch % self.loss_print == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                      str(epoch),
                      ', Loss: ',
                      str(np.round(np.mean(self.loss_history_V[-self.loss_trail:]), 3)))

        # set V in evaluation mode
        self.V.eval()


    # update the policy according to a batch of trajectories (actor)
    def update_policy(self,
                      Ntrajectories,  # number of trajectories
                      Mtransitions,  # number of transitions
                      batch_size=50,  # batch size for the update
                      Nepochs=100,  # number of epochs
                      rng_seed=None):  # random seed
        # print progress
        print('--Update of pi--')
        batch_size = np.minimum(batch_size, Ntrajectories)

        # set the policy in training mode
        self.policy.train()

        # Define the exponential learning rate scheduler
        gamma = 0.9
        step_size = 10
        scheduler = optim.lr_scheduler.StepLR(optimizer=self.policy.optimizer, step_size=step_size, gamma=gamma)


        for epoch in range(Nepochs):
            T.autograd.set_detect_anomaly(True)

            # zero grad
            self.policy.zero_grad()

            # sample a batch of transitions
            trajs, transitions = self.sim_trajectories(batch_size,
                                                       Mtransitions,
                                                       choose='random',
                                                       seed=rng_seed)

            # value function of the next time step
            obx_tp1 = T.stack((transitions["x_tp1"].clone(),
                               transitions["y_tp1"].clone(),
                               transitions["timestep_tp1"].clone()), -1)
            V_tp1 = self.V(obx_tp1.clone()).squeeze()

            # combine both the cost and value function & calculate the gradient of the value function
            V_loss = self.risk_measure.get_V_loss(transitions["cost_t_a"].detach()
                                               + np.exp(-self.params["r"]) * V_tp1.detach(),
                                               transitions["cost_t_b"].detach(),
                                               transitions["log_prob_t"],
                                               trajs["mortality"].detach().unsqueeze(1).repeat(1, Mtransitions, 1))

            # loss for each initial state
            # grad_loss = T.nan_to_num(V_loss, nan=1e-5)  # + self.params["lam"] * (A_std + I_std)
            loss = T.mean(V_loss)
            loss.requires_grads = True

            # check for NaN or -inf values in total_loss
            if T.isnan(loss).any() or T.isinf(loss).any():
                print('V_loss = ', V_loss)
                print('loss = ', loss.item())
                raise ValueError("Loss is NaN or Inf before backward.")

            # optimization step
            loss.to(self.device).backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            # policy gradient
            loss_wp = T.mean(loss)
            loss_wp.to(self.device).backward()
            self.policy.optimizer.step()

            # store the loss
            self.loss_history_policy.append(loss.detach().cpu().numpy())

            # store the loss gradients
            self.gradients.append([param.grad.clone() for param in self.policy.parameters() if param.grad is not None])

            # print progress
            if epoch % self.loss_print == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                      str(epoch),
                      ', Loss: ',
                      str(np.round(np.mean(self.loss_history_policy[-self.loss_trail:]), 4)))

            # adjust the learning rate
            scheduler.step()

        # set the policy in evaluation mode
        self.policy.eval()

    # plot the strategy at any point in the algorithm
    def plot_current_policy(self, time_idx):
        # find the best actions
        hist2dim_A = np.zeros([len(self.env.spaces["x_space"]), len(self.env.spaces["y_space"])])
        hist2dim_I = np.zeros([len(self.env.spaces["x_space"]), len(self.env.spaces["y_space"])])
        for x_idx, x_val in enumerate(self.env.spaces["x_space"]):
            for y_idx, y_val in enumerate(self.env.spaces["y_space"]):
                hist2dim_A[len(self.env.spaces["x_space"]) - x_idx - 1, y_idx], hist2dim_I[len(self.env.spaces["x_space"]) - x_idx - 1, y_idx], _ = \
                    self.select_actions(T.Tensor([x_val]).unsqueeze(1).to(self.device),
                                        T.Tensor([y_val]).unsqueeze(1).to(self.device),
                                        T.tensor([time_idx]).unsqueeze(1).to(self.device),
                                        'best')

        # plot the policy (A, I)
        plt.imshow(hist2dim_A,
                   interpolation='none',
                   cmap=utils.cmap,
                   extent=[np.min(self.env.spaces["y_space"]),
                           np.max(self.env.spaces["y_space"]),
                           np.min(self.env.spaces["x_space"]),
                           np.max(self.env.spaces["x_space"])],
                   aspect='auto',
                   vmin=0.0,
                   vmax=1.0)
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Best actions of amount A; Time step:' + str(time_idx + 1))
        plt.xlabel("Income")
        plt.ylabel("Wealth")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + '-'.join(self.method) +
                    '/time' + str(time_idx + 1) +
                    '/best_action_A_timestep' + str(time_idx + 1) +
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=False)
        plt.clf()

        plt.imshow(hist2dim_I,
                   interpolation='none',
                   cmap=utils.cmap,
                   extent=[np.min(self.env.spaces["y_space"]),
                           np.max(self.env.spaces["y_space"]),
                           np.min(self.env.spaces["x_space"]),
                           np.max(self.env.spaces["x_space"])],
                   aspect='auto',
                   vmin=0.0,
                   vmax=1.0)
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Best actions of insurance I; Time step:' + str(time_idx + 1))
        plt.xlabel("Income")
        plt.ylabel("Wealth")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + '-'.join(self.method) +
                    '/time' + str(time_idx + 1) +
                    '/best_action_I_timestep' + str(time_idx + 1) +
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=False)
        plt.clf()

    # plot the entire strategy at any point in the algorithm
    def plot_current_policies(self):
        for time_idx in self.env.spaces["t_space"][:-1][::-1]:
            self.plot_current_policy(time_idx)

    # plot the value function at any point in the algorithm
    def plot_current_V(self, time_idx):
        # find the best actions
        hist2dim = np.zeros([len(self.env.spaces["x_space"]), len(self.env.spaces["y_space"])])
        for x_idx, x_val in enumerate(self.env.spaces["x_space"]):
            for y_idx, y_val in enumerate(self.env.spaces["y_space"]):
                obs = T.stack((x_val * T.ones(1), y_val * T.ones(1), time_idx * T.ones(1)), -1)
                hist2dim[len(self.env.spaces["x_space"]) - x_idx - 1, y_idx] = self.V(obs)

        # plot the value function
        plt.imshow(hist2dim,
                   interpolation='none',
                   cmap=utils.cmap,
                   extent=[np.min(self.env.spaces["y_space"]),
                           np.max(self.env.spaces["y_space"]),
                           np.min(self.env.spaces["x_space"]),
                           np.max(self.env.spaces["x_space"])],
                   aspect='auto')
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Value function; Time step:' + str(time_idx + 1))
        plt.xlabel("Income")
        plt.ylabel("Wealth")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()

        # Create the directory if it doesn't exist
        os.makedirs(self.repo, exist_ok=True)

        # save the figure
        plt.savefig(self.repo + '/' + '-'.join(self.method) +
                    '/time' + str(time_idx + 1) +
                    '/V_function_timestep' + str(time_idx + 1) +
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=False)
        plt.clf()

    # plot the entire value function at any point in the algorithm
    def plot_current_Vs(self):
        for time_idx in self.env.spaces["t_space"][:-1][::-1]:
            self.plot_current_V(time_idx)

    ## functions to obtain the optimal policy and value function
    # obtain (discrete) state from a (continuous) observation
    def get_state(self, x, y):
        x_bin = np.digitize(x, self.env.spaces["x_space"])
        x_bin[x_bin == 0] = 1
        y_bin = np.digitize(y, self.env.spaces["y_space"])

        return x_bin, y_bin

    # give the set of valid actions
    def get_valid_actions(self, x, y):
        condition_A = np.max(np.abs(self.env.spaces["A_space"]-0.5)) <= 0.5
        condition_I = np.abs(x + self.env.params["alpha"]*y) >= np.max(self.env.spaces["I_space"])
        valid_A = self.env.spaces["A_space"][condition_A]
        valid_I = self.env.spaces["I_space"][condition_I]

        return valid_A, valid_I

