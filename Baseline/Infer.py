import numpy as np
import AirEnv as Air
from Agent import Agent
from Databatch import Databatch
from SaveLoad import save_model, load_model
import Tools

import torch
from scipy.interpolate import interp1d
import panel2D as pan

import os
import time

torch.autograd.set_detect_anomaly(True)

import logging

# logs setting
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logger = logging.getLogger("infer_logger")
logger.setLevel(logging.INFO)

log_file = os.path.join(log_directory, 'advance_v3_infer.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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

    Tools.plot_histogram(steps_array, target_lift_coefficient, num_samples)
    mean_value, min_value, max_value = Tools.calculate_statistics(steps_array)
    logger.info(f"average step: {mean_value}, min step: {min_value}, max step: {max_value}")
    
    return results

def inferring_for_more_target(clts, agent, num_samples):
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
    # set the parameters
    # set the parameters
    args = Tools.Hyperparameters()
    seed = 10
    env = Air.AirfoilEnv(target_lift_coefficient=1.5)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    agent = Agent(args, episode_num=1000)
    
    # set the target lift coefficient for the inference stage
    clts = [0.5, 0.6, 0.7, 0.8, 0.9]
    num_samples = 50
    # load the model and test it in different environments
    
    # load the model from the checkpoint file
    load_model(agent.policy_net, agent.value_net, './checkpoint/ckpt_999_ranmod_9999_.pth')
    
    inferring_for_more_target(clts, agent, num_samples)