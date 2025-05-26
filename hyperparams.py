"""
Hyperparameters
Initialization of all hyperparameters

"""

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
                 'sigma': 0.3,  # standard deviation of the stock process  ## 0.2
                 'alpha': 0.1,  # exogeneous contribution rate
                 'T': 45,  # time horizon
                 'Ndt': 45,  # number of periods (annually updated in Ex1),45*12 for monthly updated I_t
                 'qx_exproj': [0.00104, 0.00111, 0.001163, 0.001248, 0.001324, 0.0014, 0.001464, 0.001607,
                               0.001665, 0.001745, 0.001864, 0.001957, 0.001993, 0.002026, 0.002178, 0.002268,
                               0.002324, 0.002481, 0.002587, 0.00272, 0.002836, 0.002957, 0.003147, 0.003255,
                               0.003402, 0.003717, 0.003951, 0.004225, 0.00457, 0.004863, 0.005285, 0.00561,
                               0.006146, 0.006714, 0.007227, 0.007904, 0.008645, 0.009307, 0.010082, 0.010871,
                               0.011761, 0.012583, 0.013558, 0.014439, 0.015294], # mortality in 2022 without projection/ improvement
                 'a_x': [-6.62199426, -6.61687971, -6.61945225, -6.60765622, -6.59638983, -6.57901862,
                         -6.5409675, -6.52255336, -6.48942038, -6.43681585, -6.38690981, -6.33444945,
                         -6.28885908, -6.22423211, -6.16225187, -6.09679528, -6.01469309, -5.96027438,
                         -5.88313122, -5.80712464, -5.72103209, -5.64596844, -5.57205361, -5.48610356,
                         -5.40088677, -5.31559183, -5.22448498, -5.15093869, -5.05508185, -4.97249524,
                         -4.88347625, -4.80718991, -4.73048845, -4.64819047, -4.57043445, -4.49358787,
                         -4.4036374, -4.33106918, -4.23967489, -4.16666297, -4.07345451, -4.00363611,
                         -3.93078573, -3.8452619, -3.77925961],
                 'b_x': [0.02282367, 0.02298923, 0.023038, 0.02293358, 0.02246776, 0.02230933,
                         0.02219665, 0.02237816, 0.02254283, 0.02231858, 0.02277343, 0.02291991,
                         0.02337001, 0.02366213, 0.0238707, 0.02388662, 0.02369002, 0.02417846,
                         0.02392762, 0.02348254, 0.02348506, 0.0232743, 0.02321446, 0.02301529,
                         0.02266504, 0.02265092, 0.0222179, 0.02268744, 0.02261229, 0.02228169,
                         0.02212534, 0.02201933, 0.02192104, 0.02122023, 0.02093824, 0.02076544,
                         0.02027868, 0.02054719, 0.02034537, 0.02016378, 0.01991855, 0.0199037,
                         0.01995882, 0.02004766, 0.019983],
                 'k_t': [-26.43815153, -26.87067208, -27.34089493, -27.65797126, -28.18839126,
                         -28.22188979, -28.77749188, -27.89277533, -27.97398361, -28.48083927,
                         -29.06078298, -29.07079435, -29.23740246, -29.37742606, -29.46007891,
                         -30.01586595, -30.44748452, -30.98763193, -30.89659569, -31.14806456,
                         -31.59391956, -32.15351702, -32.60929649, -33.05802916, -33.19730399,
                         -33.43571589, -33.27913032, -33.64470855, -34.18342727, -34.342581,
                         -34.44040008, -34.74344078, -34.82654182, -35.17799369, -35.50952786,
                         -35.9025991, -36.47570428, -37.01889113, -37.58990321, -37.82938428,
                         -38.28350292, -38.62493151, -38.40100622, -38.88580053, -39.28886451],
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
                  'lr_pi': 1e-7,  # learning rate of the neural net associated with pi
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
