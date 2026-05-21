from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    def opposite_direction(self):
        match self:
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT
    
    # returns col, row
    def to_coords(self, coords):
        c = coords[0]
        r = coords[1]

        match self:
            case Direction.UP:
                return [c, c + 1], [r, r]
            case Direction.DOWN:
                return [c, c + 1], [r + 1, r + 1]
            case Direction.LEFT:
                return [c, c], [r, r + 1]
            case Direction.RIGHT:
                return [c + 1, c + 1], [r, r + 1]

class Action(Enum):
    MOVE = 0
    CLIMB = 1