from my_env.MyEnv import MyEnv
import numpy as np
from my_env.Direction import Direction
from my_env.Keyboard import get_direction
from my_env.Config import InputFormat, InputMode, Difficulty

def initiate_env(input_format, no_walls):
    flattened_input = input_format==InputFormat.FLAT

    env = MyEnv(
        grid_width=12,
        grid_height=5,
        goal=np.array([10, 2]),
        flattened_input = flattened_input
        )

    env.add_danger(coordinates=(9,2))
    env.add_danger(coordinates=(9,3))
    env.add_danger(coordinates=(5,4))

    env.add_bonus(coordinates=(1,0))
    env.add_bonus(coordinates=(6,2))

    if no_walls:
        return env

    env.add_wall(coordinates=[0,2], direction=Direction.RIGHT)

    env.add_wall(coordinates=[3,0], direction=Direction.RIGHT)
    env.add_wall(coordinates=[3,1], direction=Direction.RIGHT)
    env.add_wall(coordinates=[3,2], direction=Direction.RIGHT)
    env.add_wall(coordinates=[3,3], direction=Direction.RIGHT)

    env.add_wall(coordinates=[4,1], direction=Direction.RIGHT)
    env.add_wall(coordinates=[4,2], direction=Direction.RIGHT)
    env.add_wall(coordinates=[4,3], direction=Direction.RIGHT)

    env.add_wall(coordinates=[5,1], direction=Direction.RIGHT)
    env.add_wall(coordinates=[5,2], direction=Direction.RIGHT)

    env.add_wall(coordinates=[6,0], direction=Direction.RIGHT)
    env.add_wall(coordinates=[6,1], direction=Direction.RIGHT)
    env.add_wall(coordinates=[6,2], direction=Direction.RIGHT)
    
    env.add_wall(coordinates=[6,2], direction=Direction.DOWN)

    return env

def get_action(mode, format):
    match mode:
        case InputMode.MANUAL:
            action = None
            while action is None:
                action = get_direction(format)
        case InputMode.RANDOM:
            action = env.action_space.sample()

    return action


# Run as a script:
# ----------------
if __name__=="__main__":

    max_num_steps = 50
    num_epochs = 10

    input_mode = InputMode.MANUAL
    difficulty = Difficulty.HARD

    input_format, no_walls = difficulty.get_config()

    for _ in range(num_epochs):
        # Create environment:
        # -------------------
        env = initiate_env(input_format, no_walls)

        state, info = env.reset()

        print("Initial_state: ", state, "Distance to goal: ", info["Distance to goal"])

        for _ in range(max_num_steps):

            action = get_action(input_mode, input_format)
            next_step, done, reward, info = env.step(action)            

            env.render()

            print(f"Next-state: {next_step}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

            if done:
                env.close()
                break

        env.close()
