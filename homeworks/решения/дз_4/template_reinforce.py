import numpy as np
import torch
import torch.nn as nn

n_actions = 2

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states, model):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :param model: torch model
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability
    with torch.no_grad():
        # Convert numpy states to torch tensor
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        # Get logits from model
        logits = model(states_tensor)
        
        # Apply softmax to get probabilities
        probs_tensor = torch.softmax(logits, dim=-1)
        
        # Convert back to numpy
        probs = probs_tensor.numpy()
    
    return probs

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    # Initialize cumulative rewards array
    cumulative_rewards = [0] * len(rewards)
    
    # Start from the last timestep and work backwards
    cumulative_reward = 0
    for t in reversed(range(len(rewards))):
        cumulative_reward = rewards[t] + gamma * cumulative_reward
        cumulative_rewards[t] = cumulative_reward
    
    return cumulative_rewards

def get_loss(logits, actions, rewards, n_actions=n_actions, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # Compute probabilities from logits
    probs = torch.softmax(logits, dim=-1)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    # Use gather to select log probabilities for the actions that were taken
    log_probs_for_actions = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    
    # Compute policy gradient objective: J_hat = mean(log_pi(a|s) * G_t)
    # Note: cumulative_returns doesn't need gradients, it's just rewards
    J_hat = torch.mean(log_probs_for_actions * cumulative_returns.detach())
    
    # Compute entropy for regularization: H = -sum(p * log(p))
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))
    
    # Final loss: we want to maximize J_hat + entropy_coef * entropy
    # So we minimize -(J_hat + entropy_coef * entropy)
    loss = -(J_hat + entropy_coef * entropy)

    return loss