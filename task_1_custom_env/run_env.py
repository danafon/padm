from engine.MyEnv import initiate_env
from engine.Direction import Direction
from engine.Keyboard import get_direction
from engine.Config import InputMode, Difficulty

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

    for _ in range(num_epochs):
        # Create environment:
        # -------------------

        env = initiate_env(
            input_format,
            no_walls,
            goal_coordinates=goal_coordinates,
            dangers=dangers,
            bonuses=bonuses,
            walls=walls,
            random_initialization=random_initialization
        )
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
