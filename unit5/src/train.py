import numpy as np

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import numpy as np

def reinforce(policy, env, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep, as
        # the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t])
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

if __name__ == '__main__':

    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

    pixelcopter_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 10000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 0.95,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    # Create policy and place it to the device
    # torch.manual_seed(50)
    pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"],
                                pixelcopter_hyperparameters["action_space"],
                                pixelcopter_hyperparameters["h_size"]).to(device)
    pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

    scores = reinforce(pixelcopter_policy,
                       env,
                       pixelcopter_optimizer,
                       pixelcopter_hyperparameters["n_training_episodes"],
                       pixelcopter_hyperparameters["max_t"],
                       pixelcopter_hyperparameters["gamma"],
                       10)

    import utils

    mean_reward, std_reward = utils.evaluate_agent(eval_env,
        pixelcopter_hyperparameters["max_t"],
        pixelcopter_hyperparameters["n_evaluation_episodes"],
        pixelcopter_policy)

    print(mean_reward, std_reward)

    # from huggingface_hub import login
    # login("XXXX")

    # repo_id = "toinsson/Reinforce-pixelcopter-0" #TODO Define your repo id {username/Reinforce-{model-id}}
    # utils.push_to_hub(repo_id, env_id,
    #     pixelcopter_policy, # The model we want to save
    #     pixelcopter_hyperparameters, # Hyperparameters
    #     eval_env, # Evaluation environment
    #     video_fps=30
    #     )


