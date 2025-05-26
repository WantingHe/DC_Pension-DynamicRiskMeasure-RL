
"""
Main -- Algorithmic Trading Problem
Value function & policy represented by a single ANN
Value function is learned from the current policy
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# local imports
import utils
import hyperparams
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import TradingEnv
from actor_critic import ActorCriticPG
# misc
import time
import pdb  # debugging
from datetime import datetime
import pandas as pd

"""
Parameters
"""

# running on a personal computer with or a GPU server
computer = 'cuda'  # 'cpu' | 'cuda'
preload = False  # load pre-trained model prior to the training phase

# risk measures used
rm_list = [['CVaR','CVaR']]
alpha_cvar = [0.2, 0.2] # threshold for the conditional value-at-risk

mortality_list = ['qx_exproj']   # 'qx_lc' | 'qx_exproj'

# parameters for the model and algorithm
repo_name, envParams, algoParams = hyperparams.initParams()

print_progress = 200  # number of epochs before printing the time/loss
plot_progress = 50  # number of epochs before plotting the policy/value function
save_progress = 100  # number of epochs before saving the policy/value function ANNs

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
hyperparams.printParams(envParams, algoParams)
print('*  alpha_cvar: ', alpha_cvar)

# create a new directory
repo = repo_name
data_repo = repo_name
utils.directory(repo)

# loop for all risk measures
for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method_a = ', method[0], ',', 'Method_b = ', method[1], ' ***\n')
    start_time = time.time()

    for method_item in method:
        # create repositories
        if(method_item == 'CVaR'):
            method_item = method_item + str(round(alpha_cvar[idx_method],3 ))


    # loop through each mortality option
    for mortality in mortality_list:
        print(f'\n*** Training with mortality option: {mortality} ***\n')
        start_time = time.time()

        # create the environment and risk measure objects
        env = TradingEnv(envParams)
        risk_measure = RiskMeasure(params=envParams,
                                   Type_a=method[0],
                                   Type_b=method[1],
                                   alpha=alpha_cvar)

        utils.directory(repo + '/' + mortality)

        if mortality == 'qx_exproj':
            mortality_rates = np.array([envParams["qx_exproj"] for _ in range(algoParams["Ntrajectories"])])
        elif mortality == 'qx_proj':
            mortality_rates = np.array([envParams["qx_proj"] for _ in range(algoParams["Ntrajectories"])])
        elif mortality == 'qx_lc':
            mortality_rates = env.LC_mortality_generator(algoParams["Ntrajectories"], num_years=envParams["Ndt"])

        # create policy & value function objects
        # single neural network; (wealth x income x time)
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
                                     mortality=mortality_rates,
                                     repo=repo,
                                     method=method,
                                     env=env,
                                     policy=policy,
                                     V=value_function,
                                     risk_measure=risk_measure,
                                     gamma=algoParams["gamma"],
                                     rng_seed=algoParams["seed"])

        if preload:
            # load the weights of the pre-trained model
            actor_critic.policy.load_state_dict(T.load(data_repo + '/' + mortality + '/policy_model.pt'))
            actor_critic.V.load_state_dict(T.load(data_repo + '/' + mortality + '/V_model.pt'))

        ## TRAINING PHASE
        # first estimate of the value function
        actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                    Mtransitions=algoParams["Mtransitions"],
                                    batch_size=algoParams["batch_V"],
                                    Nepochs=algoParams["Nepochs_V_init"],
                                    rng_seed=algoParams["seed"])


        for epoch in range(algoParams["Nepochs"]):
            # estimate the value function of the current policy
            actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                        Mtransitions=algoParams["Mtransitions"],
                                        batch_size=algoParams["batch_V"],
                                        Nepochs=algoParams["Nepochs_V"],
                                        rng_seed=algoParams["seed"])

            # update the policy by policy gradient
            actor_critic.update_policy(Ntrajectories=algoParams["Ntrajectories"],
                                        Mtransitions=algoParams["Mtransitions"],
                                        batch_size=algoParams["batch_pi"],
                                        Nepochs=algoParams["Nepochs_pi"],
                                        rng_seed=algoParams["seed"])

            # print progress
            if epoch % print_progress == 0 or epoch == algoParams["Nepochs"] - 1:
                print('*** Epoch = ', str(epoch) ,
                        ' completed, Duration = ', "{:.3f}".format(time.time() - start_time), ' secs ***')
                start_time = time.time()

            # save progress
            if epoch % save_progress == 0:
                now = datetime.now()
                # save the neural network
                T.save(actor_critic.policy.state_dict(),
                        repo + '/' + mortality + '/policy_model' + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.pt')
                T.save(actor_critic.V.state_dict(),
                        repo + '/' + mortality + '/V_model' + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.pt')

        # save the neural network
        T.save(actor_critic.policy.state_dict(),
                repo + '/' + mortality + '/policy_model.pt')
        T.save(actor_critic.V.state_dict(),
                repo + '/' + mortality + '/V_model.pt')
        # to load the model, M = ModelClass(*args, **kwargs); M.load_state_dict(T.load(PATH))

        # save the training losses
        train_loss_V.extend(actor_critic.loss_history_V)
        train_loss_policy.extend(actor_critic.loss_history_policy)

        # save tabular data as CSV
        tabular_V = pd.DataFrame(train_loss_V, columns=['loss'])
        tabular_V.to_csv(repo + '/' + mortality + '/training_loss_V.csv',
                            index=False)
        tabular_policy = pd.DataFrame(train_loss_policy, columns=['loss'])
        tabular_policy.to_csv(repo + '/' + mortality + '/training_loss_policy.csv',
                            index=False)

        # print progress
        print('*** Training phase completed! ***')

        ###  Graph 1: Training Loss Function of V and policy
        # draw training loss graph for Value NN
        plt.plot(range(1, len(train_loss_V) + 1, 50), [train_loss_V[i] for i in range(1, len(train_loss_V) + 1, 50)])
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        # plt.title("Training loss of V")
        plt.tight_layout()
        plt.savefig(repo + '/' + mortality + '/training_loss_V.eps')
        plt.clf()

        # draw training loss graph for policy NN
        plt.plot(range(1, len(train_loss_policy)), [train_loss_policy[i] for i in range(1, len(train_loss_policy))])
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        # plt.title("Training loss of policy")
        plt.tight_layout()
        plt.savefig(repo + '/' + mortality + '/training_loss_policy.eps')
        plt.clf()

