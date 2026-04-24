from my_env.MyEnv import MyEnv
import numpy as np
from my_env.Direction import Direction
from my_env.Keyboard import get_direction
from enum import Enum

def initiate_env():
    env = MyEnv(grid_width=12, grid_height=5, goal=np.array([10, 2]))

    env.add_danger(coordinates=(9,2))
    env.add_danger(coordinates=(9,3))
    env.add_danger(coordinates=(5,4))

    env.add_bonus(coordinates=(1,0))
    env.add_bonus(coordinates=(6,2))

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


class Mode(Enum):
    MANUAL = 0
    RANDOM = 1

def get_action(mode):
    match mode:
        case Mode.MANUAL:
            dirr = None
            while dirr is None:
                dirr = get_direction()
            action = [0, dirr]
        case Mode.RANDOM:
            action = env.action_space.sample()

    return action


# Run as a script:
# ----------------
if __name__=="__main__":

    max_num_steps = 50
    num_epochs = 10

    mode = Mode.MANUAL

    for _ in range(num_epochs):
        # Create environment:
        # -------------------
        env = initiate_env()

        state, info = env.reset()

        print("Initial_state: ", state, "Distance to goal: ", info["Distance to goal"])

        for _ in range(max_num_steps):

            next_step, done, reward, info = env.step(get_action(mode))            

            env.render()

            print(f"Next-state: {next_step}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

            if done:
                env.close()
                break

        env.close()
