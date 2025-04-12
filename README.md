# DRL-based-hydrofoil-design

This is the code repository of the paper: **A Deep Reinforcement Learning Based Design Method for the Hydrofoil Shape**. The code used for baseline and four experiments is in here.

In Baseline, we storage all basic code we used in our study:

- **main.py**: contains the codes for training and evaulation process.
- **panel2D.py**: is a computing programming for 2D hydrofoil parameters.
- **AirEnv.py**: is the DRL environemnt for the baseline agent, containing the defination of reward, state, action, termination and their updating.
- **Agent.py**: containg the defination of policy nets, value nets, agent, and their updating. The core logic of PPO is in this part.
- **InterPolate.py**: contains the code for computing $\rho_1, \rho_2, \rho_3$ and other middle parameters during the update of rewards and states.
- **Infer.py**: contatins two functions for testing the ability of the agent.
- **Databatch.py**: is the defination of the bach in PPO, for storge the data in training for updating the policy and value nets.
- **SaveLoad.py**: is the functions to save and load the model during and after finushing the training.
- **Tools.py**: contains some tools in our study:
  - `check_and_create_dir()`: check and create all folders we needs before the training.
  - `Hyperparameters()`: setting all hyperparameters in PPO.
  - `plot_reward()`: plot the learning curve after finushing each stage of training.
  - `plot_histogram()`: plot the histogram of the steps the agent used during the testing.
  - `calculate_statistics()`: compute the maximum, minimum and mean of the steps the agent used during the testing.
  - `generate_env()`: generate the $Cl_{target}$ for the Stage 3 of traing.
