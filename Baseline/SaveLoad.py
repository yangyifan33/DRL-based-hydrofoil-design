import torch
import os

def save_model(agent, episode, batch, reward_vec, f_scaling_coe,
              save_path="model_checkpoint_v2_moretarget.pth"):
    '''
    save the model and training state
    '''
    
    torch.save({
        'episode': episode,
        'reward_vec': reward_vec,  # save reward vector
        'f_scaling_coe': f_scaling_coe,  # save scaling coefficient generator
        'policy_state_dict': agent.policy_net.state_dict(),
        'value_state_dict': agent.value_net.state_dict(),
        'optimizer_policy_state_dict': agent.optimizer_policy.state_dict(),
        'optimizer_value_state_dict': agent.optimizer_value.state_dict(),

        'gamma': agent.gamma,
        'epsilon': agent.epsilon,
        'lamda': agent.lamda,

        'batch_size': agent.batch_size,
        'mini_batch_size': agent.mini_batch_size,
        'epochs': agent.epochs,

        'lr_a': agent.lr_a,
        'lr_c': agent.lr_c,

        # save batch data
        'batch_states': batch.state, 
        'batch_actions': batch.action,
        'batch_logprob': batch.logprob,
        'batch_rewards': batch.reward,
        'batch_next_state': batch.next_sate,
        'batch_dw': batch.dw,
        'batch_done': batch.done,
        'batch_count': batch.count,
    }, save_path)
    print(f"Checkpoint saved at episode {episode} with {batch.count} pending samples")

def load_model(agent, batch ,load_path="model_checkpoint_v2_moretarget.pth"):
    """load the model and training state"""
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        
        # recover the net parameters
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.value_net.load_state_dict(checkpoint['value_state_dict'])
        
        # recover the optimizer parameters
        agent.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        agent.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
        
        # recover the hyperparameters
        agent.gamma = checkpoint['gamma']
        agent.epsilon = checkpoint['epsilon']
        agent.lamda = checkpoint['lamda']

        agent.batch_size = checkpoint['batch_size']
        agent.mini_batch_size = checkpoint['mini_batch_size']
        agent.epochs = checkpoint['epochs']

        agent.lr_a = checkpoint['lr_a']
        agent.lr_c = checkpoint['lr_c']

        # recover the batch data
        batch.state = checkpoint['batch_states']
        batch.action = checkpoint['batch_actions']
        batch.logprob = checkpoint['batch_logprob']
        batch.reward = checkpoint['batch_rewards']
        batch.next_sate = checkpoint['batch_next_state']
        batch.dw = checkpoint['batch_dw']
        batch.done = checkpoint['batch_done']
        batch.count = checkpoint['batch_count']

        
        resume_data = {
            'episode': checkpoint['episode'],
            'reward_vec': checkpoint['reward_vec'],
            'f_scaling_coe': checkpoint['f_scaling_coe']
        }
        
        print(f"Loaded checkpoint from episode {resume_data['episode']}")
        print(f"Pending batch data: {batch.count} samples")
        return resume_data
    else:
        print("No checkpoint found, starting fresh")
        return {
            'episode': 0,
            'reward_vec': [],
            'f_scaling_coe': None
        }