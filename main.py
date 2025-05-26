"""
Algorithmic Trading with Risk-Sensitive Reinforcement Learning

This script implements an actor-critic reinforcement learning algorithm for
optimal life insurance purchasing and investment strategies. The model considers
different mortality projections and risk measures (CVaR).

Key Components:
- Policy and value function approximation using neural networks
- Risk-sensitive reinforcement learning with CVaR
- Mortality projection options (Lee-Carter model (with exogeneous estimated parameters) vs static mortality)
- Training and testing phases with visualization

Pretrained models are included for immediate visualization. For full training:
1. Execute main_train.py (requires GPU for efficient training)
2. Generate plots via main.py
"""

"""
Plots -- Algorithmic Trading Problem
Value function & policy represented by a single ANN
Value function is learned from the current policy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# numpy
import numpy as np
import numpy.matlib
# plotting
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
# pytorch
import torch as T
import torch.optim as optim
# personal files
import utils
import hyperparams
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import TradingEnv
from actor_critic import ActorCriticPG
from scipy import stats
from scipy.stats import multivariate_normal
# misc
import time
import pandas as pd
from tabulate import tabulate
import pdb  # use with set_trace() for the debugger

"""
Parameters
"""

# running on a personal computer or a Compute server
computer = 'cpu'  # 'cpu' | 'cuda'

# risk measures used
rm_list = [['CVaR','CVaR']]

# cohort mortality list
mortality_list = ['qx_exproj', 'qx_lc']

# parameters for the model and algorithm
repo_name, envParams, algoParams = hyperparams.initParams()

seed = 1  # set seed for replication purposes

# testing phase parameters
Nsimulations = 5000  # number of simulations following the optimal strategy

# font sizes for figures
plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)


"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
hyperparams.printParams(envParams, algoParams)

# create a new directory
if (computer == 'cpu'):  # personal computer
    repo = repo_name
if (computer == 'cuda'):  # Compute group server
    repo = repo_name

utils.directory(repo)

# selected timsteps
print_list = [0, 22, 44]

for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # initiate matrix to store all testing trajectories
    costs_a = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    costs_b = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    A_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    I_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    purchasing_power = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    x_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    y_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))

    Wt_collection = np.zeros((envParams["Ndt"], len(mortality_list)))
    Bt_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))
    FCt_collection = np.zeros((Nsimulations, envParams["Ndt"], len(mortality_list)))

    # create the environment and risk measure objects
    env = TradingEnv(envParams)
    risk_measure = RiskMeasure(params=envParams,
                               Type_a='CVaR',
                               Type_b='CVaR')

    for idx_mortality, mortality in enumerate(mortality_list):
        print(f'\n*** Training with mortality option: {mortality} ***\n')

        if mortality == 'qx_exproj':
            q_xt = np.array([envParams["qx_exproj"] for _ in range(Nsimulations)])
        elif mortality == 'qx_proj':
            q_xt = np.array([envParams["qx_proj"] for _ in range(Nsimulations)])
        elif mortality == 'qx_lc':
            q_xt = env.LC_mortality_generator(Nsimulations, num_years=envParams["Ndt"])


        # create policy & value function objects
        policy = PolicyApprox(3, env,
                              n_layers=algoParams["layers_pi"],
                              hidden_size=algoParams["hidden_pi"],
                              learn_rate=algoParams["lr_pi"])
        value_function = ValueApprox(3, env,
                                     n_layers=algoParams["layers_V"],
                                     hidden_size=algoParams["hidden_V"],
                                     learn_rate=algoParams["lr_V"])

        # initialize the actor-critic algorithm
        actor_critic = ActorCriticPG(params=envParams,
                                     mortality=q_xt,
                                     repo=repo,
                                     method=method,
                                     env=env,
                                     policy=policy,
                                     V=value_function,
                                     risk_measure=risk_measure,
                                     gamma=algoParams["gamma"],
                                     rng_seed=algoParams["seed"])


        # load the trained model
        actor_critic.policy.load_state_dict(T.load(repo + '/' + mortality + '/policy_model.pt',
                                                   weights_only=True, map_location=T.device(computer)))
        actor_critic.V.load_state_dict(T.load(repo + '/' + mortality + '/V_model.pt',
                                              weights_only=True, map_location=T.device(computer)))

        # print progress
        print('*** Training phase completed! ***')


        ## TESTING PHASE
        # set seed for reproducibility purposes
        T.manual_seed(seed)
        np.random.seed(seed)
        print('seed = ', seed)

        # initialize the starting state
        x, y = env.reset(Nsimulations)

        # resimulate trajectories for the trained model and save the numerical results
        for timestep in env.spaces["t_space"]:
            # simulate transitions according to the policy
            A, I, _ = actor_critic.select_actions(x.unsqueeze(-1).repeat(1, 1).to(computer),
                                                   y.unsqueeze(-1).repeat(1, 1).to(computer),
                                                   timestep * T.ones(Nsimulations).unsqueeze(-1).repeat(1, 1).to(computer),
                                                   'best')

            # store instantaneous wealth and purchasing power
            x_collection[:, timestep, idx_mortality] = x.detach().cpu().numpy()
            y_collection[:, timestep, idx_mortality] = y.detach().cpu().numpy()
            purchasing_power[:, timestep, idx_mortality] = (x + envParams["alpha"] * y).detach().cpu().numpy()

            # update state variables and costs by step function
            x, y, cost_a, cost_b = env.step(x.to(computer), y.to(computer), T.tensor(q_xt[:,timestep]).to(computer), A, I)

            # store costs
            costs_a[:, timestep, idx_mortality] = cost_a.detach().cpu().numpy()
            costs_b[:, timestep, idx_mortality] = cost_b.detach().cpu().numpy()

            # store strategy
            A_collection[:, timestep, idx_mortality] = A.detach().cpu().numpy()
            I_collection[:, timestep, idx_mortality] = I.detach().cpu().numpy()



        ### Graph 2: 3D - Bivariate Normal Distribution (A_t, I_t) for selected timesteps

        tab_data = []
        for idx_plot, t in enumerate(print_list):
            # store mean of the state variables among the generated N trajectories at t = timestep
            x_mean = np.mean(x_collection[:, t, idx_mortality])
            y_mean = np.mean(y_collection[:, t, idx_mortality])

            # observations as a formatted tensor
            exp_obx_t = T.stack((x_mean * T.ones(1).unsqueeze(-1), y_mean * T.ones(1).unsqueeze(-1), t * T.ones(1).unsqueeze(-1)), -1)

            # obtain parameters of the distribution
            A_mean, A_std, I_mean, I_std, correlation = policy(exp_obx_t.clone().to(computer),
                                                                 x_mean * T.ones(1).unsqueeze(-1).to(computer) + envParams["alpha"] * y_mean * T.ones(1).unsqueeze(-1).to(computer))


            mean = np.squeeze(np.array([A_mean.detach().squeeze(-1).cpu().numpy(),
                                        I_mean.detach().squeeze(-1).cpu().numpy()]))
            cov = np.array([[A_std.item()**2, (correlation * A_std * I_std).item()], [(correlation * A_std * I_std).item(), I_std.item()**2]])
            jitter = 1e-5  # Small positive value
            cov = cov + jitter

            rv = multivariate_normal(mean, cov)

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Compute the 2D kernel density estimate
            X2, Y2 = np.meshgrid(np.linspace(max(0, A_mean.detach().item() - 4*A_std.detach().item()),
                                           A_mean.detach().item() + 4*A_std.detach().item(), 100),
                               np.linspace(max(0, I_mean.detach().item() - 4*I_std.detach().item()),
                                           I_mean.detach().item() + 4*I_std.detach().item(), 100))

            # Probability Density
            pos = np.empty(X2.shape + (2,))
            pos[:, :, 0] = X2
            pos[:, :, 1] = Y2
            Z2 = rv.pdf(pos)

            # Plot the 3D surface
            surf = ax.plot_surface(X2, Y2, Z2,
                                   rstride=1, cstride=1,
                                   cmap='viridis',
                                   alpha=0.6,
                                   color=utils.colors[idx_plot],
                                   edgecolor='none')

            # ax.set_title(f"t = {t}")
            ax.set_xlabel(r"$\alpha^\ast_t$", fontsize=16, labelpad=15)
            ax.set_ylabel(r"$I^\ast_t$", fontsize=16, labelpad=15)
            ax.set_zlabel("Density", fontsize=16, labelpad=15)
            plt.tick_params(axis='both', labelsize=12)
            plt.tick_params(axis='z', labelsize=12)

            # Adjust the layout and save the figure
            # plt.suptitle(" - Distribution of the Optimal Strategy (A, I) with E[X] ="
            #               + str("{:.2f}".format(x_mean)) + " E[Y] = " + str("{:.2f}".format(y_mean)))
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.tight_layout()
            plt.savefig(repo + '/' + mortality + '/fixed_optimal_A_I_3d_t' + str(t) + '.pdf', transparent=True, dpi=600)
            plt.clf()

            # Data: save the data for the figure
            data_dict = {
                'X': X2.flatten(),
                'Y': Y2.flatten(),
                'Z': Z2.flatten(),
                'parameters': {
                    't': t,
                    'x_mean': x_mean,
                    'y_mean': y_mean,
                    'A_mean': A_mean.item(),
                    'I_mean': I_mean.item(),
                    'A_std': A_std.item(),
                    'I_std': I_std.item(),
                    'correlation': correlation.item()
                }
            }
            np.savez(repo + '/' + mortality + '/fixed_optimal_A_I_3d_t' + str(t) + '_data.npz',
                     **data_dict)

            # Table: present optimal policy distribution parameters for selected timesteps
            tab_data.append([t, x_mean, y_mean, A_mean.item(), I_mean.item(), A_std.item(), I_std.item(), correlation.item()])

        headers = ['Timestep', 'x_mean', 'y_mean', 'A_mean', 'I_mean', 'A_std', 'I_std', 'correlation']
        print('Mortality option: ' + mortality)
        print(tabulate(tab_data, headers=headers, tablefmt="grid"))



### Graph 3: Dynamic Optimal Policy (mean + CI)

        A_mean = np.zeros(envParams["Ndt"])
        A_std = np.zeros(envParams["Ndt"])
        I_mean = np.zeros(envParams["Ndt"])
        I_std = np.zeros(envParams["Ndt"])
        V_est = np.zeros(envParams["Ndt"])
        x_dyn_mean = np.zeros(envParams["Ndt"])
        y_dyn_mean = np.zeros(envParams["Ndt"])
        pi_dyn = np.zeros((envParams["Ndt"], Nsimulations))
        beta_dyn = np.zeros((envParams["Ndt"], Nsimulations))

        tab_dyn = []

        # X, Y, policy all N trajectories and then take the mean
        for t in range(envParams["Ndt"]):
            # Store mean of the state variables among the generated N trajectories at t = timestep
            x_dyn_mean[t] = np.mean(x_collection[:, t, idx_mortality])
            y_dyn_mean[t] = np.mean(y_collection[:, t, idx_mortality])

            # Set expected observations as a formatted tensor
            exp_dyn_t = T.stack((T.tensor(x_collection[:, t, idx_mortality]).unsqueeze(-1), T.tensor(y_collection[:, t, idx_mortality]).unsqueeze(-1),
                                 t * T.ones(Nsimulations).unsqueeze(-1)), -1)

            # Obtain parameters of the optimal policy distribution at t
            A_t, A_t_std, I_t, I_t_std, correlation = policy(exp_dyn_t.clone().to(computer).to(T.float32),
                                                                       (T.tensor(x_collection[:, t, idx_mortality])
                                                                        + envParams["alpha"] * T.tensor(y_collection[:, t, idx_mortality])).unsqueeze(-1).to(computer).to(T.float32))
            v_t = value_function(exp_dyn_t.clone().to(computer).to(T.float32)).squeeze()

            # Obtain proportions of the optimal strategy at t
            pi_t = A_t.squeeze(-1) / (T.tensor(x_collection[:, t, idx_mortality]) + envParams["alpha"] * T.tensor(y_collection[:, t, idx_mortality])).to(computer).to(T.float32)
            beta_t = I_t.squeeze(-1) / (T.tensor(x_collection[:, t, idx_mortality]) + envParams["alpha"] * T.tensor(y_collection[:, t, idx_mortality])).to(computer).to(T.float32)


            A_mean[t] = T.mean(A_t).item()
            A_std[t] = T.mean(A_t_std).item()
            I_mean[t] = T.mean(I_t).item()
            I_std[t] = T.mean(I_t_std).item()
            V_est[t] = T.mean(v_t).item()
            pi_dyn[t, :] = pi_t.cpu().detach().numpy()
            beta_dyn[t, :] = beta_t.cpu().detach().numpy()

            tab_dyn.append([t, x_dyn_mean[t], y_dyn_mean[t], A_mean[t], I_mean[t], A_std[t], I_std[t], V_est[t]])


        headers = ['Timestep', 'x_mean', 'y_mean', 'A_mean', 'I_mean', 'A_std', 'I_std', 'Estimated_V']
        print(tabulate(tab_dyn, headers=headers, tablefmt="grid"))


        # Calculate the 90% CIs
        lower_ci_A = np.maximum(A_mean - 1.645 * A_std, 0)
        upper_ci_A = A_mean + 1.645 * A_std
        lower_ci_I = np.maximum(I_mean - 1.645 * I_std, 0)
        upper_ci_I = I_mean + 1.645 * I_std

        pi_mean = np.zeros(envParams["Ndt"])
        beta_mean = np.zeros(envParams["Ndt"])
        lower_ci_pi = np.zeros(envParams["Ndt"])
        upper_ci_pi = np.zeros(envParams["Ndt"])
        lower_ci_beta = np.zeros(envParams["Ndt"])
        upper_ci_beta = np.zeros(envParams["Ndt"])

        for t in range(envParams["Ndt"]):
            pi_mean[t] = np.mean(pi_dyn[t, :])
            beta_mean[t] = np.mean(beta_dyn[t, :])
            pi_sem = stats.sem(pi_dyn[t, :])
            beta_sem = stats.sem(beta_dyn[t, :])

            df = Nsimulations - 1

            lower_ci_pi[t], upper_ci_pi[t] = stats.t.interval(0.9, df, loc=pi_mean[t], scale=pi_sem)
            lower_ci_beta[t], upper_ci_beta[t] = stats.t.interval(0.9, df, loc=beta_mean[t], scale=beta_sem)

            lower_ci_pi[t] = np.nan_to_num(lower_ci_pi[t], nan=0.99*pi_mean[t])
            lower_ci_beta[t] = np.nan_to_num(lower_ci_beta[t], nan=0.99 * beta_mean[t])
            upper_ci_pi[t] = np.nan_to_num(upper_ci_pi[t], nan=1.01 * pi_mean[t])
            upper_ci_beta[t] = np.nan_to_num(upper_ci_beta[t], nan=1.01 * beta_mean[t])

            # save the plot data
            x_ticks = np.arange(envParams["Ndt"]) + 22
            plot_data = {
                'A': {
                    'mean': A_mean,
                    'std': A_std,
                    'lower_ci': lower_ci_A,
                    'upper_ci': upper_ci_A
                },
                'I': {
                    'mean': I_mean,
                    'std': I_std,
                    'lower_ci': lower_ci_I,
                    'upper_ci': upper_ci_I
                },
                'pi': {
                    'mean': pi_mean,
                    'lower_ci': lower_ci_pi,
                    'upper_ci': upper_ci_pi,
                    'all_values': pi_dyn
                },
                'beta': {
                    'mean': beta_mean,
                    'lower_ci': lower_ci_beta,
                    'upper_ci': upper_ci_beta,
                    'all_values': beta_dyn
                },
                'state_variables': {
                    'x_mean': x_dyn_mean,
                    'y_mean': y_dyn_mean
                },
                'value_function': V_est,
                'parameters': envParams,
                'age_labels': x_ticks.tolist(),
                'visible_labels': print_list,
                'label_indices': print_list
            }

            # save the complete data structure
            np.savez(repo + '/' + mortality + '/dynamic_optimal_policy_data.npz', **plot_data)

            # save tabular data as CSV
            tabular_data = pd.DataFrame(tab_dyn, columns=[
                'Timestep', 'x_mean', 'y_mean',
                'A_mean', 'I_mean',
                'A_std', 'I_std',
                'Estimated_V'
            ])
            tabular_data.to_csv(repo + '/' + mortality + '/dynamic_optimal_policy_table.csv',
                                index=False)

        # create the figure
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 6))

        # plot the mean of the optimal policy and 90% CI for Dimension 1
        ax1.plot(A_mean, label='Mean of Optimal Alpha', color=utils.colors[idx_mortality])
        ax1.plot(lower_ci_A, '--', alpha=0.6, color=utils.colors[idx_mortality], label='90% CI (Alpha)')
        ax1.plot(upper_ci_A, '--', alpha=0.6, color=utils.colors[idx_mortality])
        ax1.fill_between(np.arange(envParams["Ndt"]), lower_ci_A, upper_ci_A, alpha=0.2, color=utils.colors[idx_mortality])
        ax1.set_xlabel('Age')
        ax1.set_ylabel(r'$\alpha_t$')
        # ax1.set_title('Optimal A and 90% CI', y=1.05, fontsize=14)
        ax1.legend()
        ax1.set_ylim([min(A_mean - 4 * A_std), max(A_mean + 4 * A_std)])

        # modify x-tick labels for ax1
        x_ticks = np.arange(envParams["Ndt"]) + 22  # Shift x-tick labels by 22
        all_labels = [str(x_ticks[i]) for i in np.arange(envParams["Ndt"])]  # create labels for all ticks
        selective_idx = np.arange(0, len(all_labels), 5)  # Indices of the labels to show
        selective_labels = [all_labels[i] for i in selective_idx]

        # set the tick labels while keeping the full tick positions
        ax1.set_xticks(np.arange(envParams["Ndt"]))
        ax1.set_xticklabels(all_labels)

        # show selective_labels only
        for i, tick in enumerate(ax1.xaxis.get_major_ticks()):
            if i not in selective_idx:
                tick.label1.set_visible(False)

        # plot the mean of the optimal policy and 90% CI for Dimension 2
        ax2.plot(I_mean, label='Mean of Optimal I', color=utils.colors[idx_mortality])
        ax2.plot(lower_ci_I, '--', alpha=0.6, color=utils.colors[idx_mortality], label='90% CI (I)')
        ax2.plot(upper_ci_I, '--', alpha=0.6, color=utils.colors[idx_mortality])
        ax2.fill_between(np.arange(envParams["Ndt"]), lower_ci_I, upper_ci_I, alpha=0.2, color=utils.colors[idx_mortality])
        ax2.set_xlabel('Age')
        ax2.set_ylabel(r'$I_t$')
        # ax2.set_title('Optimal I and 90% CI', y=1.05, fontsize=14)
        ax2.legend()
        ax2.set_ylim([min(I_mean - 4 * I_std), max(I_mean + 4 * I_std)])

        # modify x-tick labels for ax2
        ax2.set_xticks(np.arange(envParams["Ndt"]))
        ax2.set_xticklabels(all_labels)

        # show selective_labels only
        for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
            if i not in selective_idx:
                tick.label1.set_visible(False)

        # plot the mean of the optimal policy and 90% CI for Dimension 3
        ax3.plot(pi_mean, label='Mean of Optimal Pi', color=utils.colors[idx_mortality])
        ax3.plot(lower_ci_pi, '--', alpha=0.6, color=utils.colors[idx_mortality], label='90% CI (Pi)')
        ax3.plot(upper_ci_pi, '--', alpha=0.6, color=utils.colors[idx_mortality])
        ax3.fill_between(np.arange(envParams["Ndt"]), lower_ci_pi, upper_ci_pi, alpha=0.2,
                         color=utils.colors[idx_mortality])
        ax3.set_xlabel('Age')
        ax3.set_ylabel(r'$\pi_t$')
        # ax3.set_title('Optimal pi and 90% CI', y=1.05, fontsize=14)
        ax3.legend()
        ax3.set_ylim([min(lower_ci_pi), max(upper_ci_pi)])

        # modify x-tick labels for ax3
        ax3.set_xticks(np.arange(envParams["Ndt"]))
        ax3.set_xticklabels(all_labels)

        # show selective_labels only
        for i, tick in enumerate(ax3.xaxis.get_major_ticks()):
            if i not in selective_idx:
                tick.label1.set_visible(False)

        # plot the mean of the optimal policy and 90% CI for Dimension 4
        ax4.plot(beta_mean, label='Mean of Optimal Beta', color=utils.colors[idx_mortality])
        ax4.plot(lower_ci_beta, '--', alpha=0.6, color=utils.colors[idx_mortality], label='90% CI (Beta)')
        ax4.plot(upper_ci_beta, '--', alpha=0.6, color=utils.colors[idx_mortality])
        ax4.fill_between(np.arange(envParams["Ndt"]), lower_ci_beta, upper_ci_beta, alpha=0.2,
                         color=utils.colors[idx_mortality])
        ax4.set_xlabel('Age')
        ax4.set_ylabel(r'$\beta_t$')
        # ax4.set_title('Optimal Beta and 90% CI', y=1.05, fontsize=14)
        ax4.legend()
        ax4.set_ylim([min(lower_ci_beta), max(upper_ci_beta)])

        # modify x-tick labels for ax4
        ax4.set_xticks(np.arange(envParams["Ndt"]))
        ax4.set_xticklabels(all_labels)

        # show selective_labels only
        for i, tick in enumerate(ax4.xaxis.get_major_ticks()):
            if i not in selective_idx:
                tick.label1.set_visible(False)

        # Save the plot
        plt.tight_layout()
        plt.savefig(repo + '/' + mortality + '/dynamic_optimal_policy.pdf',
                    transparent=True, dpi=600)
        plt.clf()


    ### Graph 4: Distribution of the terminal wealth
    # consider the case that the individual remain alive till T
    wealth_T = x_collection[:, -1, :] + envParams["alpha"] * y_collection[:, -1, :] - costs_a[:, -1, :]
    legend_list = ['mortality without projection', 'mortality with LC projection']

    # save data for reformatting
    wealth_dict = {legend_list[0]: wealth_T[:, 0], legend_list[1]: wealth_T[:, 1]}
    np.savez(repo + '/terminal_wealth.npz', **wealth_dict)

    # set a grid for the histogram
    grid = np.linspace(np.min(wealth_T), min(np.max(wealth_T), 180), 100)

    plt.figure(figsize=(10, 6))

    for idx_plot in range(wealth_T.shape[1]):
        # set a grid for the histogram
        # grid = np.linspace(np.min(wealth_T), min(np.max(wealth_T), 200), 100)

        # plot the histogram for each mortality
        plt.hist(x=wealth_T[:, idx_plot],
                 alpha=0.4,
                 # bins=grid,
                 color=utils.colors[idx_plot],
                 density=True,
                 bins='auto')

        plt.xlabel(r"$W_T$", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        # plt.title("Distribution of the terminal wealth")

        # for idx_mortality, mortality in enumerate(mortality_list):
        # plot gaussian KDEs
        kde = gaussian_kde(wealth_T[:, idx_plot], bw_method='silverman')
        plt.plot(grid,
                 kde(grid),
                 color=utils.colors[idx_plot],
                 linewidth=1.5,
                 label=legend_list[idx_plot])
        # plot quantiles of the distributions
        plt.axvline(x=np.quantile(wealth_T[:, idx_plot], 0.05),
                    linestyle='dashed',
                    color=utils.colors[idx_plot],
                    linewidth=1.0)
        plt.axvline(x=np.mean(wealth_T[:, idx_plot]),
                    linestyle='dotted',
                    color=utils.colors[idx_plot],
                    linewidth=1.0)
        plt.axvline(x=np.quantile(wealth_T[:, idx_plot], 0.95),
                    linestyle='dashed',
                    color=utils.colors[idx_plot],
                    linewidth=1.0)

    plt.xlim(0, 150)
    plt.legend()
    plt.tight_layout()
    plt.savefig(repo + '/comparison_terminal_wealth.pdf',
                transparent=True, dpi=600)
    plt.clf()

    ### Graph 5: Plots of mortality settings
    # call mortality rate lists
    qx_exproj = envParams["qx_exproj"]
    qx_lc = np.mean(env.LC_mortality_generator(1), axis=0)

    # create a range for the x-axis
    ages = list(range(22, len(qx_exproj)+22))

    # create the figure
    plt.figure(figsize=(8, 6))

    # plot each mortality rate on the same axes
    plt.plot(ages, qx_exproj, label='Mortality Rate (2022)', marker='o', color=utils.mblue)
    plt.plot(ages, qx_lc, label='Mortality Rate (LC projection 2022-2066)', marker='o', color=utils.mred)

    # adding titles and labels
    plt.xlabel('Age', fontsize=14)
    plt.ylabel(r'$\ln(m_{x,t})$', fontsize=14)
    # plt.ylabel('Mortality Rate', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid()

    # customize x-ticks
    plt.xticks(ages)
    all_labels = [str(age) for age in ages]
    selective_idx = np.arange(0, len(all_labels), 5)
    plt.xticks(ticks=[ages[i] for i in selective_idx], labels=[all_labels[i] for i in selective_idx])

    # save the plot
    plt.tight_layout()  # adjust layout to prevent overlap
    plt.savefig(repo + '/Mortality_comparison.pdf', transparent=True, dpi=600)
    plt.clf()


# print progress
print('*** Testing phase completed! ***')






