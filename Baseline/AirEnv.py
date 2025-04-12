import gym
import numpy as np
import gym.utils.seeding as seeding

import panel2D as pan
import interPolate

# 环境定义
class AirfoilEnv(gym.Env):
    def __init__(self, target_lift_coefficient):
        super(AirfoilEnv, self).__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([5, 0.1, 1.5]), 
            dtype=np.float32)  # the range of AoA, camber and lift coefficient
        self.action_space = gym.spaces.Box(
            low=np.array([-0.75, -0.02]), 
            high=np.array([0.75, 0.02]), 
            dtype=np.float32)  # max change of AoA and camber
        self.target_lift_coefficient = target_lift_coefficient
        self.current_step = 0

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, random_state=None):
        self.current_step = 0   
        if random_state:    # if use random inital state
            AoA = np.random.uniform(2.5, 5)
            camber = np.random.uniform(0.05, 0.1)
            self.state = np.array([AoA, camber, self.target_lift_coefficient], dtype=np.float32)  # inital AoA and camber
        else:
            self.state = np.array([5, 0.1, self.target_lift_coefficient], dtype=np.float32) # inital AoA and camber
        return self.state

    def step(self, action):
        self.current_step += 1

        # update AoA and camber
        front_action =  np.clip(action, self.action_space.low, self.action_space.high)
        front_state = np.clip(self.state[:2] + front_action, self.observation_space.low[:2], self.observation_space.high[:2])
        
        # check the shape of self.state
        assert self.state.shape == (3,), f"Expected self.state shape (6.3), but got {self.state.shape}"
        # compute lift coefficient and pressure distribution by using panel method
        lift_coefficient, pressure_distribution = self.calculate_aerodynamic_properties(front_state)

    # compute the parameters about the pressure distribution
        I = interPolate.calinterPlate(pressure_distribution)

        # compute the minimum value of pressure distribution
        pressure_min = I.compute_min()
    
        # compute the difference stability of the upper and lower edge of the central area
        middle_parameter = I.compute_middle_parameter()

        # compute the degree of front-end warping
        value_is_injected = I.is_injected()

        # update the state
        self.state = np.array([front_state[0], front_state[1], self.target_lift_coefficient], dtype=np.float32)

        co = [-25, -6, 15, -5, 3]
        # compute the reward
        reward = (
            co[0] * abs(self.target_lift_coefficient - lift_coefficient) + 
            co[1] * (pressure_min[0] / pressure_min[1] - 1) +  
            co[2] * value_is_injected +
            co[3] * abs(middle_parameter) + 
            co[4] *  pressure_min[0]
        )
        
        # judege whether the episode is done
        done = ((abs(lift_coefficient - self.target_lift_coefficient) < 0.01) and  
                pressure_min[0] / pressure_min[1] <= 1.05 and 
                abs(middle_parameter) <= 0.6 and 
                value_is_injected >= -0.01)

        return self.state, reward, done, {}

    def step_for_infer(self, action): #step for inference,remove the computation and updating for reward 

        self.current_step += 1

        # undate AoA and camber
        front_action =  np.clip(action, self.action_space.low, self.action_space.high)
        front_state = np.clip(self.state[:2] + front_action, self.observation_space.low[:2], self.observation_space.high[:2])

        assert self.state.shape == (3,), f"Expected self.state shape (6.3), but got {self.state.shape}"
        
        # compute new lift coefficient and pressure distribution by using panel method
        lift_coefficient, pressure_distribution = self.calculate_aerodynamic_properties(front_state)

    # compute the parameters about the pressure distribution
        I = interPolate.calinterPlate(pressure_distribution)

        # compute the minimum value of pressure distribution
        pressure_min = I.compute_min()
    
        # compute the difference stability of the upper and lower edge of the central area
        middle_parameter = I.compute_middle_parameter()

        # compute the degree of front crossover
        value_is_injected = I.is_injected()

        self.state = np.array([front_state[0], front_state[1], self.target_lift_coefficient], dtype=np.float32)
        
        # judege whether the episode is done by termination conditions
        done = ((abs(lift_coefficient - self.target_lift_coefficient) < 0.01) and  
                pressure_min[0] / pressure_min[1] <= 1.05 and 
                abs(middle_parameter) <= 0.6 and 
                value_is_injected >= -0.01
        )

        return self.state, self.current_step ,done, {}

    def calculate_aerodynamic_properties(self, state):
        angle_in_degree = state[0]
        camber_ratio = state[1]
        s = pan.solver()
        s.solve(angleInDegree=angle_in_degree, camberRatio=camber_ratio)
        
        lift_coefficient = s._Cl
        pressure_distribution = s._CpArray
                
        return lift_coefficient, pressure_distribution
