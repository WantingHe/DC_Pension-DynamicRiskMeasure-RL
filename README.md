# DC_Pension-DynamicRiskMeasure-RL
This is the code accompanying the paper *"Periodic Evaluation of DC Pension Fund: A Dynamic Risk Measure Approach"* by Wanting He, Wenyuan Li, and Yunran Wei.
This study adapts a risk-sensitive RL framework for optimal contribution/insurance decisions under mortality risk.

**Disclaimer:** I cleaned up the code a bit before release. The code presents the numerical cases with/without mortality projection within periodic evaluation framework. A few test runs indicate it still works, but if you encounter problems, please let me know via email [u3006949@connect.hku.hk](u3006949@connect.hku.hk). Also, if there are questions or anything unclear, please don't hesitate to approach me - feedback is very welcome!

## 🗂 Repository Structure  
```
DCPension-DRM/
├── pretrained_models/               # Saved model checkpoints
│   ├── policy_model.pt              # Trained policy network
│   └── V_model.pt                   # Value function approximator
│
└── src/                             # Core implementation
    ├── actor_critic.py              # RL algorithm
    ├── envs.py                      # Pension fund environment
    ├── models.py                    # NN architectures
    ├── risk_measure.py              # CVaR calculations
    ├── main_train.py                # Training script
    └── main.py                      # Evaluation & plotting
```
