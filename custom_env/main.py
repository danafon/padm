# Imports:
# --------
from engine.MyEnv import initiate_env
from q_learning import train_q_learning, visualize_q_table
from engine.Direction import Direction
from engine.Config import InputMode, Difficulty

# User definitions:
# -----------------
train = True
visualize_results = True
render = False

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  # If True, the agent will be initialized randomly in the environment

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 20_000  # Number of episodes

input_mode = InputMode.MANUAL
difficulty = Difficulty.MEDIUM

input_format, no_walls = difficulty.get_config()

random_initialization = True
goal_coordinates = (10, 2)
random_initialization = True
dangers = ((9,2), (9,3), (5,4))
bonuses = ((1,0), (6,2))
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
    train_q_learning(env=env,
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
    visualize_q_table(danger_state_coordinates=dangers,
                      goal_coordinates=goal_coordinates,
                      wall_coordinates=walls,
                      bonus_coordinates=bonuses,
                      q_values_path="q_table.npy")
