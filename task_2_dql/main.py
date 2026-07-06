# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from pathlib import Path
from env import ContinuousMazeEnv
import numpy as np

from time import sleep

# User definitions:
# -----------------
train_dqn = False
test_dqn = True
render = True #do not turn on for training

#! Define env attributes (environment specific)
dim_actions = 4
dim_states = 2

exp_no = 6

# Hyperparameters:
# ----------------
# learning_rate = 0.005
learning_rate = 0.0003

gamma = 0.98

buffer_limit = 50_000
batch_size = 128

num_episodes = 20_000

max_steps = 400

epsilon_decay = 0.9995

# Main:
# -----
if train_dqn:
    if render:
        env = ContinuousMazeEnv(render_mode='human')
    else:
        env = ContinuousMazeEnv(render_mode='auto')

    #! Initialize the Q Net and the Q Target Net
    q_net = Qnet(dim_actions=dim_actions, 
                 dim_states=dim_states)
    q_target = Qnet(dim_actions=dim_actions, 
                    dim_states=dim_states)
    
    q_target.load_state_dict(q_net.state_dict())

    #! Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 50
    episode_reward = 0.0

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate, weight_decay=1e-5)

    rewards = []

    epsilon = 1
    min_epsilon = 0.02
    memory_size = 2000

    for n_epi in range(num_episodes):
        #! Epsilon decay (Please come up with your own logic) todo
        # epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))  # ! Linear annealing from 8% to 1%
        if memory.size() > memory_size:
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

        s, _ = env.reset()
        done = False

        #! Define maximum steps per episode, here 10,000
        for _ in range(max_steps):
            #! Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon, dim_actions)
            
            s_prime, r, done, _, _ = env.step(a)

            done_mask = 0.0 if done else 1.0

            #! Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r

            if done:
                break

        if memory.size() > memory_size:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        #! Define a stopping condition for the game: todo
        if np.mean(rewards[-10:]) > 9:
            break
    
    env.close()

    #! Save the trained Q-net
    torch.save(q_net.state_dict(), Path.cwd() / f"dqn_{exp_no}.pth")

    #! Plot the training curve
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(Path.cwd() / f"training_curve_{exp_no}.png")
    plt.show()


# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    if render:
        env = ContinuousMazeEnv(render_mode='human')
    else:
        env = ContinuousMazeEnv(render_mode='auto')

    dqn = Qnet(dim_actions=dim_actions, 
               dim_states=dim_states)
    dqn.load_state_dict(torch.load(Path.cwd() / f"dqn_{exp_no}.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            #! Completely exploit while testing
            if render:
                env.render()
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _, _ = env.step(action.argmax().item())
            s = s_prime
            # sleep(0.1)

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
