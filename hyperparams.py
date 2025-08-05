"""
Hyperparameters
Initialization of all hyperparameters

"""
import pandas as pd
qx_df = pd.read_csv('mortality_rates.csv')


# initialize parameters for the environment and algorithm
def initParams():
    # name of the repository
    repo_name = 'Ex01'

    # parameters for the model
    envParams = {'x0': 5,  # initial wealth
                 'y0': 60,  # initial income
                 'mu_y': 0.03,  # risky growth rate of the income process
                 'sigma_y': 0.05,  # standard deviation of the income process
                 'r': 0.02,  # risk-free interest rate
                 'mu': 0.15,  # risky return rate of the stock process
                 'sigma': 0.2,  # standard deviation of the stock process  
                 'alpha': 0.1,  # exogeneous contribution rate
                 'T': 45,  # time horizon
                 'Ndt': 45,  # number of periods (annually updated in Ex1),45*12 for monthly updated I_t
                 'qx_exproj': qx_df['qx_exproj'].values,
                 'qx_lc': qx_df['qx_lc'].values,
                 'rho': 0.3755}  # correlation coefficient between BM of stock and salary

    # parameters for the algorithm
    algoParams = {'Ntrajectories': 750,   # number of generated trajectories # 1000  ## 500
                  'Mtransitions': 500,  # number of additional transitions for each state ## 500 #50
                  'Nepochs': 100,  # number of epochs of the whole algorithm # 100 ## 300
                  'gamma': 1.00,  # discount factor
                  'Nepochs_V_init': 100,  # number of epochs for the estimation of V during the first epoch ## 500
                  'Nepochs_V': 50,  # number of epochs for the estimation of V
                  'lr_V': 1e-4,  # learning rate of the neural net associated with V
                  'batch_V': 200,  # number of trajectories for each mini-batch in estimating V ## 200
                  'hidden_V': 32,  # number of hidden nodes in the neural net associated with V
                  'layers_V': 3,  # number of layers in the neural net associated with V
                  'Nepochs_pi': 10,  # number of epoch for the update of pi
                  'lr_pi': 0.5*1e-7,  # learning rate of the neural net associated with pi
                  'batch_pi': 100,  # number of trajectories for each mini-batch when updating pi ## 200
                  'hidden_pi': 32,  # number of hidden nodes in the neural net associated with pi
                  'layers_pi': 3,  # number of layers in the neural net associated with pi
                  'Nsims_optimal': 1000,  # number of simulations when using the brute force method
                  'seed': 1}  # set seed for replication purposes

    return repo_name, envParams, algoParams


def printParams(envParams, algoParams):
    print('** Individual assumption ** \\'
          ' T: ', envParams["T"],
          ' Ndt: ', envParams["Ndt"],
          ' x_0: ', envParams["x0"],
          ' y_0: ', envParams["y0"])
    print('** Market assumption ** \\'
          ' sigma_x: ', envParams["sigma"],
          ' mu_x: ', envParams["mu"],
          ' r: ', envParams["r"],
          ' sigma_y: ', envParams["sigma_y"],
          ' mu_y: ', envParams["mu_y"],
          ' alpha: ', envParams["alpha"])
    print('** Algorithm setting ** \\'
          '*  Ntrajectories: ', algoParams["Ntrajectories"],
          ' Mtransitions: ', algoParams["Mtransitions"],
          ' Nepochs: ', algoParams["Nepochs"],
          ' Nsims_optimal: ', algoParams["Nsims_optimal"])
    print('*  Nepochs_V_init: ', algoParams["Nepochs_V_init"],
          ' Nepochs_V: ', algoParams["Nepochs_V"],
          ' lr_V: ', algoParams["lr_V"],
          ' batch_V: ', algoParams["batch_V"],
          ' hidden_V: ', algoParams["hidden_V"],
          ' layers_V: ', algoParams["layers_V"])
    print('*  Nepochs_pi: ', algoParams["Nepochs_pi"],
          ' lr_pi: ', algoParams["lr_pi"],
          ' batch_pi: ', algoParams["batch_pi"],
          ' hidden_pi: ', algoParams["hidden_pi"],
          ' layers_pi: ', algoParams["layers_pi"])
