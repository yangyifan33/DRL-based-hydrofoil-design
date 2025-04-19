# DRL-based-hydrofoil-design

This is the code repository of the paper: **A Deep Reinforcement Learning Based Design Method for the Hydrofoil Shape** [1]. The code used for baseline and four experiments is in the **src**.

We storage all basic code we used in our study:

- **main.py**: contains the codes for training and evaulation process.
- **panel2D.py**: is a computing programming for 2D hydrofoil parameters.
- **AirEnv.py**: is the DRL environemnt for the baseline. experiment 1 and 2 agent, containing the defination of reward, state, action, termination and their updating.
- **Agent.py**: containg the defination of policy nets, value nets, agent, and their updating. The core logic of PPO is in this part.
- **InterPolate.py**: contains the code for computing $\rho_1, \rho_2, \rho_3$ and other middle parameters during the update of rewards and states.
- **Infer.py**: contatins two functions for testing the ability of the agent.
- **Databatch.py**: is the defination of the bach in PPO, for storge the data in training for updating the policy and value nets.
- **SaveLoad.py**: is the functions to save and load the model during and after finushing the training.
- **Tools.py**: contains some tools in our study:
  - `Generate_env()`: generate the environment under different experiment sittings. 
  - `check_and_create_dir()`: check and create all folders we needs before the training.
  - `Hyperparameters()`: setting all hyperparameters in PPO and setting of experiments.
  - `Hyperparameters()`: setting all hyperparameters in PPO and settings in our 5 experiments.
  - `plot_reward()`: plot the learning curve after finushing each stage of training.
  - `plot_histogram()`: plot the histogram of the steps the agent used during the testing.
  - `calculate_statistics()`: compute the maximum, minimum and mean of the steps the agent used during the testing.
  - `generate_target_Cl()`: generate the $Cl_{target}$ for the Stage 3 of traing.
When running different experiments, just modify the last 5 parameters of `Hyperparameters()`.

This project partially refers to the PPO implementation from:
Lizhi, "ppo-pytorch", GitHub repository: https://github.com/Lizhi-sjtu/ppo-pytorch
Licensed under the MIT License. These code is in **third_party**

[1] Yang, Y., & Wang, Y. (n.d.). A Deep Reinforcement Learning Based Design Method for the Hydrofoil Shape.
