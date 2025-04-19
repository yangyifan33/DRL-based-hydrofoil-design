import numpy as np
import AirEnv as Air
from Agent import Agent
from Databatch import Databatch
from SaveLoad import save_model, load_model
import Tools
import Infer

import torch
from scipy.interpolate import interp1d
import panel2D as pan

import os
import time

torch.autograd.set_detect_anomaly(True)

import logging

log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logger = logging.getLogger("main_logger")
logger.setLevel(logging.INFO)

log_file = os.path.join(log_directory, 'advance_v3.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train(args, agent, num_episodes=1000, env_random_mod = 0, load_point = 0):
    '''
    train the agent in one environment randomization model, and save the model
        :param agent: the agent to be trained
        :param num_episodes: the number of episodes to be trained
        :param save_interval: the interval to save the model
        :param env_random_mod: the random mode of the environment
        :param env_random_mod_load: the random mode of the environment to load
        :param load_point: the point to load the model
    '''
    env = Tools.Generate_env(args, target_lift_coefficient=1.5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

# Load the model
    load_path = './checkpoint/ckpt_%d_ranmod_%d_.pth' % (load_point, env_random_mod-1)
    batch = Databatch(args.batch_size, state_dim, action_dim)
    resume_data = load_model(agent, batch, load_path=load_path)

    start_episode = 0
    reward_vec = []  # vector to store rewards
    f_scaling_coe = resume_data['f_scaling_coe']
    step_for_every_10episode = 0

    # Initialize the agent: learning rate reset and episode reset
    agent.lr_reset()
    agent.episodes_reset(num_episodes)
    
    # # Initalize the reward scalar at the beginning of the training
    # if start_episode == 0 and env_random_mod == 0:
    #     print("\n--- Running environment parameter sampling ---")
    #     param_values = np.linspace(0, 1.5, 10) # uniform sampling of target lift coefficient
    #     std_values = []

    #     for param in param_values:
    #         init_env = Air.AirfoilEnv(target_lift_coefficient=param)
    #         all_rewards = []
    #         for _ in range(2000):
    #             state = init_env.reset()
    #             done = False
    #             action = init_env.action_space.sample()  # sample random action
    #             next_state, reward, done, _ = init_env.step(action)
    #             all_rewards.append(reward)
    #             if done:
    #                 break
            
    #         # compute the standard deviation of rewards
    #         if len(all_rewards) > 1:
    #             reward_std = np.std(all_rewards)
    #             reward_std = 1.0 if np.isclose(reward_std, 0) else reward_std
    #         else:
    #             reward_std = 1.0
    #         std_values.append(reward_std)
    #         print(f"Param: {param:.2f}, Reward Std: {reward_std:.4f}")
    #     # create interpolation function for reward scaling
    #     f_scaling_coe = interp1d(param_values, std_values, kind='linear', fill_value="extrapolate")
    #     print("\n--- Reward scaling interpolation function created ---\n")
    
# main training loop
    # Experiment 5: Optimize the random frequency
    hold = 0
    for episode in range(start_episode, num_episodes):
        if env_random_mod == 0: # steady environment
            target_lift_coefficient = 0.75
        elif env_random_mod == 1: # limited random environment
            target_lift_coefficient = np.random.uniform(0.5, 1.0)
            if hold == args.Random_frequancy - 1:   #optimize the random frequancy
                target_lift_coefficient = np.random.uniform(0.5, 1.0)
                hold = 0
            else:
                hold += 1
        elif env_random_mod == 2:
            target_lift_coefficient = Tools.generate_target_Cl()
            if hold == args.Random_frequancy - 1:   #optimize the random frequancy
                target_lift_coefficient = Tools.generate_target_Cl()
                hold = 0
            else:
                hold += 1 
        elif env_random_mod == 3: # environment with different target lift coefficient in 0-1.5
            target_lift_coefficient = np.random.uniform(0, 1.5)
            if hold == args.Random_frequancy - 1:   #optimize the random frequancy
                target_lift_coefficient = np.random.uniform(0, 1.5)
                hold = 0
            else:
                hold += 1

        #scaling_coe = f_scaling_coe(target_lift_coefficient) # get the scaling coefficient
        episode_rewards = []
        env = Tools.Generate_env(args, target_lift_coefficient=target_lift_coefficient) # generate the environment according to the hyperparameters

        if episode < num_episodes * 0.75: # 75% of the time use the steady start point, the rest is the random start point
            state = env.reset(random_state=False)
        else:
            state = env.reset(random_state=True)
        done = False
        
        # collect data in one epsiode
        for _ in range(args.max_steps): #The max steps in one episode is 200
            # choose action and get next state
            action, log_prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            # preprocess the reward
            #reward = reward / scaling_coe
            episode_rewards.append(reward)
            if done and env.current_step != 200:
                dw = True
            else:
                dw = False

            batch.store(state, action, log_prob, reward, next_state, dw, done)
            state = next_state
            # update the agent when the batch is full
            if batch.count == args.batch_size:
                agent.update(batch, episode)
                batch.count = 0 # clear the batch
                print("batch updated")
                logger.info(f"batch updated at episode {episode}")

            if done:
                break
        
        step_for_every_10episode += env.current_step
        reward_mean = np.sum(episode_rewards) / env.current_step
        reward_vec.append(reward_mean)
        
        if episode % 10 == 0:
            average_step = step_for_every_10episode / 10
            step_for_every_10episode = 0
            print(f"Episode {episode}, Reward: {reward_mean}, Average Step: {average_step}")
            logger.info(f"Episode {episode}, Reward: {reward_mean}, Average Step: {average_step}")

        # save the model every save_interval episodes
        if (episode % args.save_interval == 0 and episode != 0) or (episode == num_episodes-1): 
            save_model(agent=agent, episode = episode, batch = batch,
                       reward_vec = reward_vec,f_scaling_coe = f_scaling_coe, save_path='./checkpoint/ckpt_%d_ranmod_%d_.pth' % (episode, env_random_mod))
    
    #Plot the reward curve at the end of each stage
    reward_np = Tools.plot_reward(reward_vec, start_episode, num_episodes, env_random_mod)
    return reward_np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    Tools.check_and_create_dir() # create folders for saving the model and logs

    # set the parameters
    args = Tools.Hyperparameters()
    seed = 10
    env = Tools.Generate_env(args, target_lift_coefficient=1.5) # generate the environment according to the setting of experiments
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    agent = Agent(args, episode_num=1000)

    print(f"training begins, time：{time.asctime()}")
    logger.info('training begins!')
    
    # Experiment 4: Remove stage wise random
    if args.Remove_stage_wise_random:
        reward_np = train(args, agent, num_episodes=4499, env_random_mod=3, load_point=0)

    else:
    # baseline: 3 training stages
        reward_np1 = train(args, agent, num_episodes=1000, env_random_mod=0, load_point=0)
        reward_np2 = train(args, agent, num_episodes=1500, env_random_mod=1, load_point=999)
        reward_np3 = train(args, agent, num_episodes=2000, env_random_mod=2, load_point=1499)
        reward_np = np.concatenate((reward_np1, reward_np2, reward_np3))
    np.save("rewards.npy", reward_np)

    #evaluate the agent in different environments
    clts = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    Infer.inferring_for_more_target(clts=clts, agent=agent, num_samples=100)
