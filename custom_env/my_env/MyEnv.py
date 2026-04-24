# Imports:
# --------
import sys
import pygame
import numpy as np
import gymnasium as gym
from .Direction import Direction, Action
from collections import defaultdict

# ORDER OF COORDINATES: (X, Y); Y IS REVERTED!!!

# Custom Environment:
# -------------------
class MyEnv(gym.Env):        

    def __init__(self, grid_width, grid_height, goal) -> None:
        super().__init__()

# move costs to configs

# add bonuses

# add jumps
        self.state = None
        self.done = False
        self.info = {}
        self.reward = 0
        self.cell_size = 30
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.goal = goal
        self.action_space = gym.spaces.MultiDiscrete((1, 4))
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self.grid_width - 1, self.grid_height - 1], dtype=np.int32),
            dtype=np.int32
        )

        self.danger_states = []
        self.bonus_states = []
        self.walls = defaultdict(set)

        # Display:
        # --------
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_width, self.cell_size*self.grid_height))

    # Method 1:
    # ---------

    def reset(self):
        self.state = np.array([0,2])
        self.done = False
        self.reward = 0

        self.distance_to_goal()

        return self.state, self.info
    
    def add_danger(self, coordinates):
        self.danger_states.append(np.array(coordinates))

    def add_bonus(self, coordinates):
        self.bonus_states.append(np.array(coordinates))

    def add_wall(self, coordinates, direction: Direction):
        coord = tuple(coordinates)
        self.walls[coord].add(direction)
        step = self.basic_step(direction=direction, state=coordinates)
        if not np.array_equal(step, np.array([0, 0])):
            coord = tuple(np.add(coordinates, step))
            self.walls[coord].add(direction.opposite_direction())

    def distance_to_goal(self):
        dist = np.sqrt((self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2)
        self.info["Distance to goal"] = dist

    # Method 2:
    # ---------
    def basic_step(self, direction: Direction, action: Action = Action.MOVE, consider_walls = False, state = None):
        if state is None:
            state = self.state

        if (
            direction==Direction.UP and state[1]==0
            or direction==Direction.DOWN and state[1]==self.grid_height-1
            or direction==Direction.RIGHT and state[0]==self.grid_width-1
            or direction==Direction.LEFT and state[0]==0
        ):
            return np.array([0, 0])
        
        if consider_walls:
            if direction in self.walls.get(tuple(state), set()):
                return np.array([0, 0])
        
        match direction:
            case Direction.UP:
                return np.array([0, -1])
            case Direction.DOWN:
                return np.array([0, 1])
            case Direction.LEFT:
                return np.array([-1, 0])
            case Direction.RIGHT:
                return np.array([1, 0])
            case _:
                return np.array([0, 0])

    def perform_goal_check(self):
        return np.array_equal(self.state, self.goal)
    
    def perform_danger_check(self):
        return any(np.array_equal(self.state, d) for d in self.danger_states)
    
    def perform_bonus_check(self):
        return any(np.array_equal(self.state, d) for d in self.bonus_states)

    def perform_checks(self):
        if self.perform_goal_check():
            self.done = True
            self.reward = 10
        elif self.perform_danger_check():
            self.done = True
            self.reward = -20
        elif self.perform_bonus_check():
            self.reward = 3
        else:
            self.reward = -0.01


    def step(self, input):
        action = Action(input[0])
        direction = Direction(input[1])

        step = self.basic_step(action=action, direction=direction, consider_walls=True)
        self.state = np.add(self.state, step)

        # Check special rules, change reward:
        # ------------------
        # Goal:
        self.perform_checks()

        # Info:
        # -----
        self.distance_to_goal()

        return self.state, self.done, self.reward, self.info

    # Method 3:
    # ---------
    def render(self):
        # Code for closing the window
        for event in pygame.event.get():
            if event==pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background:
        # -----------
        self.screen.fill((255,255,255))

        # Draw gridlines:
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                grid = pygame.Rect(col*self.cell_size,
                                   row*self.cell_size,
                                   self.cell_size,
                                   self.cell_size)
                pygame.draw.rect(self.screen,
                                 (200,200,200),
                                 grid,
                                 1)
                
        # Draw goal:
        goal = pygame.Rect(self.goal[0]*self.cell_size,
                    self.goal[1]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        pygame.draw.rect(self.screen,
                        (64,224,208),
                        goal)


        # Add danger states:
        for each_danger in self.danger_states:
            danger = pygame.Rect(each_danger[0]*self.cell_size,
                        each_danger[1]*self.cell_size,
                        self.cell_size,
                        self.cell_size)
            pygame.draw.rect(self.screen,
                            (240,128,128),
                            danger)
            
        # add bonuses
        for each_bonus in self.bonus_states:
            bonus = pygame.Rect(each_bonus[0]*self.cell_size,
                        each_bonus[1]*self.cell_size,
                        self.cell_size,
                        self.cell_size)
            pygame.draw.rect(self.screen,
                            (243,218,88),
                            bonus)
            
        # add walls:
        for coord, directions in self.walls.items():
            for direction in directions:
                match direction:
                    case Direction.UP:
                        start_pos = (coord[0] * self.cell_size, coord[1] * self.cell_size)
                        end_pos = ((coord[0] + 1) * self.cell_size, coord[1] * self.cell_size)
                    case Direction.DOWN:
                        start_pos = (coord[0] * self.cell_size, (coord[1] + 1) * self.cell_size)
                        end_pos = ((coord[0] + 1) * self.cell_size, (coord[1] + 1) * self.cell_size)
                    case Direction.LEFT:
                        start_pos = (coord[0] * self.cell_size, coord[1] * self.cell_size)
                        end_pos = (coord[0] * self.cell_size, (coord[1] + 1) * self.cell_size)
                    case Direction.RIGHT:
                        start_pos = ((coord[0] + 1) * self.cell_size, coord[1] * self.cell_size)
                        end_pos = ((coord[0] + 1) * self.cell_size, (coord[1] + 1) * self.cell_size)
                
                pygame.draw.line(self.screen,
                                (139,69,19),
                                start_pos,
                                end_pos,
                                4)
            
        # Draw agent:
        agent = pygame.Rect(self.state[0]*self.cell_size,
                    self.state[1]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        pygame.draw.rect(self.screen,
                        (70,130,180),
                        agent)
        

        pygame.time.wait(100)
        pygame.display.flip()


    # Method 4:
    # ---------
    def close(self):
        pygame.quit()