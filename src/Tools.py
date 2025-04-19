import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d
import AirEnv as Air
import os

def check_and_create_dir():
    folder_names = ['checkpoint', 'rewards_graph', 'logs', 'step_hist']
    workspace = os.getcwd()
    for folder_name in folder_names:
        folder_path = os.path.join(workspace, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder has been created：{folder_path}")
        else:
            print(f"Folder exists：{folder_path}")

def Hyperparameters():
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_steps", type=int, default=200, help=" Maximum number of training episodes")
    parser.add_argument("--save_interval", type=int, default=50, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--epochs", type=int, default=4, help="PPO parameter")

    #seeting for the experiments
    parser.add_argument("--Use_support_dimensions", type=bool, default=False, help="Experiment 1: Use support dimensions")
    parser.add_argument("--Remove_target_Cl", type=bool, default=False, help="Experiment 2: Remove target Cl")
    parser.add_argument("--Use_all_equal_rewards_coe", type=bool, default=False, help="Experiment 3: Use all equal rewards coefficients")
    parser.add_argument("--Remove_stage_wise_random", type=bool, default=False, help="Experiment 4: Remove stage wise random")
    parser.add_argument("--Random_frequancy", type=int, default=10, help="Experiment 5: Optimize the random frequency")
    
    args = parser.parse_args()
    return args
def Generate_env(args, target_lift_coefficient):
    '''
    generate the environment according to the setting of experiments, 
    experiment 1 and 2 using different AirfoilEnv class
    expeirment 3 and baseline using the same AirfoilEnv class, but different reward coefficients
    experiment 4 and 5 using the same AirfoilEnv class, but different training process
    '''
    if args.Use_support_dimensions:
        env = Air.AirfoilEnvSD(target_lift_coefficient=target_lift_coefficient)    
    elif args.Remove_target_Cl:
        env = Air.AirfoilEnv2D(target_lift_coefficient=target_lift_coefficient)
    else:
        env = Air.AirfoilEnv(target_lift_coefficient=target_lift_coefficient, all_equal_rewards_coe=args.Use_all_equal_rewards_coe)
    return env

def plot_reward(reward_vec, start_episode, num_episodes, env_random_mod):
    reward_np = np.array(reward_vec)
    episodes_np = np.arange(start_episode, num_episodes)
    reward_pathname = str(f'./rewards_graph/{num_episodes - 1}-{env_random_mod}.jpg')
    if(num_episodes != start_episode):
        plt.ion()
        plt.plot(episodes_np, reward_np)
        plt.pause(5)
        plt.savefig(reward_pathname)
        plt.close()
    return reward_np

def plot_histogram(data, target_lift_coefficient, num_samples):
    """
    plot the histogram of the step data
    """
    step_pathname = str(f'./step_hist/{target_lift_coefficient}-sample{num_samples}.jpg')
    plt.ion()
    plt.hist(data, bins=10, edgecolor='black')
    plt.xlabel('steps')
    plt.ylabel('frenquency')
    plt.title(f'The frenquency distribution of steps for Cl = {target_lift_coefficient}')

    plt.pause(5)
    plt.savefig(step_pathname)
    plt.close()

def calculate_statistics(data):
    """
    calculate the mean, min and max of the step data

    return:
    tuple: average, min and max of the data
    """
    mean_value = np.mean(data)
    min_value = np.min(data)
    max_value = np.max(data)
    return mean_value, min_value, max_value

def generate_target_Cl():
    '''
    generate a random environment for the training stage 3, 
    where the environment will be new environemt with probability 0.8 and
    the old environment in satage 2 with probability 0.2
    '''
    a = np.random.uniform(0,1)
    if 0 < a < 0.4:
        return np.random.uniform(0, 0.5)
    if 0.4 < a < 0.6:
        return np.random.uniform(0.5, 1)
    if 0.6 < a < 1:
        return np.random.uniform(1, 1.5)
