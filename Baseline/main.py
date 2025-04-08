import numpy as np
import AirEnv as Air
from Agent import Agent
from Databatch import Databatch
from SaveLoad import save_model, load_model

import torch
from scipy.interpolate import interp1d
import panel2D as pan

import os
import time

torch.autograd.set_detect_anomaly(True)

import logging
import matplotlib.pyplot as plt

# logs setting
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = os.path.join(log_directory, 'advance_v3.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train(agent, num_episodes=1000, save_interval=50, batch_size=2048, env_random_mod = 0, env_random_mod_load = 0, load_point = 0):
    '''
    train the agent in one environment randomization model, and save the model
        :param agent: the agent to be trained
        :param num_episodes: the number of episodes to be trained
        :param save_interval: the interval to save the model
        :param env_random_mod: the random mode of the environment
        :param env_random_mod_load: the random mode of the environment to load
        :param load_point: the point to load the model
    '''
    env = Air.AirfoilEnv(target_lift_coefficient=1.5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

# Load the model if env_random_mod_load is not 0
    load_path = './checkpoint/ckpt_%d_ranmod_%d_.pth' % (load_point, env_random_mod_load)
    batch = Databatch(batch_size, state_dim, action_dim)
    resume_data = load_model(agent, batch, load_path=load_path)

    start_episode = 0
    reward_vec = []  # vector to store rewards
    f_scaling_coe = resume_data['f_scaling_coe']
    step_for_every_10episode = 0

    # Initialize the agent: learning rate reset and episode reset
    agent.lr_reset()
    agent.episodes_reset(num_episodes)
    
    # Initalize the reward scalar
    if start_episode == 0 and env_random_mod == 0:
        print("\n--- Running environment parameter sampling ---")
        param_values = np.linspace(0, 1.5, 10) # uniform sampling of target lift coefficient
        std_values = []

        for param in param_values:
            init_env = Air.AirfoilEnv(target_lift_coefficient=param)
            all_rewards = []
            for _ in range(2000):
                state = init_env.reset()
                done = False
                action = init_env.action_space.sample()  # sample random action
                next_state, reward, done, _ = init_env.step(action)
                all_rewards.append(reward)
                if done:
                    break
            
            # compute the standard deviation of rewards
            if len(all_rewards) > 1:
                reward_std = np.std(all_rewards)
                reward_std = 1.0 if np.isclose(reward_std, 0) else reward_std
            else:
                reward_std = 1.0
            std_values.append(reward_std)
            print(f"Param: {param:.2f}, Reward Std: {reward_std:.4f}")
        # create interpolation function for reward scaling
        f_scaling_coe = interp1d(param_values, std_values, kind='linear', fill_value="extrapolate")
        print("\n--- Reward scaling interpolation function created ---\n")
    hold = 0

# main training loop
    for episode in range(start_episode, num_episodes):
        if env_random_mod == 0: # steady environment
            target_lift_coefficient = 0.75
        elif env_random_mod == 1: # limited random environment
            target_lift_coefficient = np.random.uniform(0.5, 1.0)
            if hold == 9:   #optimize the random frequancy, environment will be changed every 10 episodes
                target_lift_coefficient = np.random.uniform(0.5, 1.0)
                hold = 0
            else:
                hold += 1
        elif env_random_mod == 2:
            target_lift_coefficient = generate_env()
            if hold == 9:
                target_lift_coefficient = generate_env()
                hold = 0
            else:
                hold += 1 
        elif env_random_mod == 3:
            target_lift_coefficient = np.random.uniform(0, 1.5)
            if hold == 9:
                target_lift_coefficient = np.random.uniform(0, 1.5)
                hold = 0
            else:
                hold += 1 

        scaling_coe = f_scaling_coe(target_lift_coefficient) # get the scaling coefficient
        episode_rewards = []
        env = Air.AirfoilEnv(target_lift_coefficient)
        if episode < num_episodes * 0.75: # 75% of the time use the steady start point, the rest is the random start point
            state = env.reset(random_state=False)
        else:
            state = env.reset(random_state=True)
        done = False
        
        # collect data in one epsiode
        for _ in range(200): #The max steps in one episode is 200
            # choose action and get next state
            action, log_prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            # preprocess the reward
            reward = reward / scaling_coe
            episode_rewards.append(reward)
            if done and env.current_step != 200:
                dw = True
            else:
                dw = False

            batch.store(state, action, log_prob, reward, next_state, dw, done)
            state = next_state
            # update the agent when the batch is full
            if batch.count == batch_size:
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
        if (episode % save_interval == 0 and episode != 0) or (episode == num_episodes-1): 
            save_model(agent=agent, episode = episode, batch = batch,
                       reward_vec = reward_vec,f_scaling_coe = f_scaling_coe, save_path='./checkpoint/ckpt_%d_ranmod_%d_.pth' % (episode, env_random_mod))
    
    #Plot the reward curve at the end of each stage
    reward_np = np.array(reward_vec)
    episodes_np = np.arange(start_episode, num_episodes)
    reward_pathname = str(f'./rewards_graph/{num_episodes - 1}-{env_random_mod}.jpg')
    if(num_episodes != start_episode):
        plt.ion()
        plt.plot(episodes_np, reward_np)
        plt.pause(5)
        plt.savefig(reward_pathname)
        plt.close()

    return reward_np, reward_pathname

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_env():
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

def infer(policy_net, target_lift_coefficient, num_samples=100):
    '''
    Evaluate the policy network in the inference stage, agent trying to design foil, and save the results.
    :param policy_net: the policy network to be evaluated
    :param target_lift_coefficient: the target lift coefficient
    :param num_samples: the number of samples to be evaluated
    '''
    policy_net.eval()  # set the model to evaluation mode
    results = []
    steps_array = []
    env = Air.AirfoilEnv(target_lift_coefficient)
    for i in range(num_samples):
        print(f"now sampling {i+1}")
        states = []
        state = env.reset()
        states.append(state)
        done = False

        while not done:    
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)     
            with torch.no_grad():
                dist = policy_net.get_dist(state)
                action = dist.sample()

            action_np = action.numpy().flatten()
            state, steps_used ,done, _ = env.step_for_infer(action_np)

        steps_array.append(steps_used)
        logger.info(f"camber angle: {state[0]}, camber ratio: {state[1]}, Using Steps: {steps_used}")
        results.append(state)

    plot_histogram(steps_array, target_lift_coefficient, num_samples)
    mean_value, min_value, max_value = calculate_statistics(steps_array)
    logger.info(f"average step: {mean_value}, min step: {min_value}, max step: {max_value}")
    
    return results

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

def inferring_test(clts, agent, num_samples):
    '''
    Test the agent in different environments and save the results.
    '''
    for target_lift_coefficient in clts:
        print(f"Inferring begins, time：{time.asctime()}，target：{target_lift_coefficient}")
        logger.info("orginal net inferring begins!")
        results = infer(agent.policy_net, target_lift_coefficient, num_samples=num_samples)

            # calculate the average AoA and camber ratio
        mean_camber_angle = np.mean([result[0] for result in results])
        mean_camber_ratio = np.mean([result[1] for result in results])

            # calculate the average lift coefficient
        s = pan.solver()
        s.solve(angleInDegree=mean_camber_angle, camberRatio=mean_camber_ratio)
        lift_coefficient = s._Cl

        print(f"target: {target_lift_coefficient}")
        print(f"averge AoA: {mean_camber_angle}")
        print(f"averge camber ratio: {mean_camber_ratio}")
        print(f"actual Cl:{lift_coefficient}")
        logger.info(f"target lift coefficient: {target_lift_coefficient}; averge AoA: {mean_camber_angle}; averge camber ratio: {mean_camber_ratio};lift coefficient:{lift_coefficient}")
            
        for i, (camber_angle, camber_ratio, _) in enumerate(results):
            print(f"sample {i + 1}: AoA = {camber_angle}, camber ratio = {camber_ratio}")

if __name__ == "__main__":
    env = Air.AirfoilEnv(target_lift_coefficient=1.5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim, gamma=0.99, lamda=0.95, 
                  batch_size=2048, mini_batch_size=64, lr_a=3e-4, lr_c=3e-4, 
                  epochs=4, epsilon=0.2, episode_num=1000)

    print(f"training begins, time：{time.asctime()}")
    logger.info('training begins!')
    # 3 training stages
    reward_np1, _ = train(agent, num_episodes=1000, save_interval=50, batch_size=2048, env_random_mod=0, env_random_mod_load=0, load_point=0)
    reward_np2, _ = train(agent, num_episodes=1500, save_interval=50, batch_size=2048, env_random_mod=1, env_random_mod_load=0, load_point=999)
    reward_np3, _ = train(agent, num_episodes=2000, save_interval=50, batch_size=2048, env_random_mod=2, env_random_mod_load=1, load_point=1499)
    #train(agent, num_episodes=1000, save_interval=50, batch_size=2048, env_random_mod=3, env_random_mod_load=2, load_point=1999)
    
    # save the rewards
    reward_np = np.concatenate((reward_np1, reward_np2, reward_np3))
    np.save("rewards.npy", reward_np)
    clts = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    inferring_test(clts=clts, agent=agent, num_samples=100)
