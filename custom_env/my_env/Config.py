from enum import Enum

class InputMode(Enum):
    MANUAL = 0
    RANDOM = 1

class InputFormat(Enum):
    FLAT = 0
    TWO_D = 1

class Difficulty(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

    def get_config(self):
        match self:
            case Difficulty.EASY:
                input_format = InputFormat.FLAT
                no_walls = True
            case Difficulty.MEDIUM:
                input_format = InputFormat.FLAT
                no_walls = False
            case Difficulty.HARD:
                input_format = InputFormat.TWO_D
                no_walls = False

        return input_format, no_walls