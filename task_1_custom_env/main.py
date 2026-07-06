# Imports:
# --------
from engine.MyEnv import initiate_env
from q_learning import train_q_learning, visualize_q_table
from engine.Direction import Direction
from engine.Config import InputMode, Difficulty
from pathlib import Path

# User definitions:
# -----------------
train = True
visualize_results = True
render = False

experiment_to_visualize = 1
"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  # If True, the agent will be initialized randomly in the environment

learning_rate = [1e-2, 3e-4, 1e-1][0]  # Learning rate (alpha)
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate for exploration
no_episodes = 10_000  # Number of episodes

input_mode = InputMode.MANUAL
difficulty = Difficulty.MEDIUM

input_format, no_walls = difficulty.get_config()

random_initialization = True
goal_coordinates = (10, 2)
random_initialization = True
dangers = ((9,2), (9,3), (5,4), (9,1), (10,3), (7,0))
bonuses = ((3,0), (6,2), (0,4))
walls = (
    ([0,2], Direction.RIGHT),
    ([3,0], Direction.RIGHT),
    ([3,1], Direction.RIGHT),
    ([3,2], Direction.RIGHT),
    ([3,3], Direction.RIGHT),
    ([4,1], Direction.RIGHT),
    ([4,2], Direction.RIGHT),
    ([4,3], Direction.RIGHT),
    ([5,1], Direction.RIGHT),
    ([5,2], Direction.RIGHT),
    ([6,0], Direction.RIGHT),
    ([6,1], Direction.RIGHT),
    ([6,2], Direction.RIGHT),
    ([6,2], Direction.DOWN),

    ([0,3], Direction.DOWN),
    ([0,3], Direction.RIGHT),
    ([0,1], Direction.RIGHT),
    ([1,0], Direction.DOWN),
    ([2,0], Direction.RIGHT),
    ([2,1], Direction.RIGHT),
    ([2,1], Direction.DOWN),
    ([2,2], Direction.RIGHT),
    ([2,2], Direction.LEFT),
    ([2,3], Direction.LEFT),
    ([2,4], Direction.RIGHT),
    ([8,0], Direction.DOWN),
    ([8,1], Direction.DOWN),
    ([8,2], Direction.DOWN),
    ([8,3], Direction.DOWN),
)


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = initiate_env(
            input_format,
            no_walls,
            goal_coordinates=goal_coordinates,
            dangers=dangers,
            bonuses=bonuses,
            walls=walls,
            random_initialization=random_initialization
        )

    # Train a Q-learning agent:
    # -------------------------
    result_path = train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     render=render)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    path = result_path if train else Path("experiments" / experiment_to_visualize)
    visualize_q_table(danger_state_coordinates=dangers,
                      goal_coordinates=goal_coordinates,
                      wall_coordinates=walls,
                      bonus_coordinates=bonuses,
                      q_values_path=result_path,
                      save_res=train)
