import pygame
from my_env.Config import InputFormat

def format_output(format, res):
    if format == InputFormat.TWO_D:
        return res
    
    return res[0] * 4 + res[1]

def get_direction(format):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return format_output(format, [0, 0])
            elif event.key == pygame.K_DOWN:
                return format_output(format, [0, 1])
            elif event.key == pygame.K_RIGHT:
                return format_output(format, [0, 2])
            elif event.key == pygame.K_LEFT:
                return format_output(format, [0, 3])
            elif event.key == pygame.K_w:
                return format_output(format, [1, 0])
            elif event.key == pygame.K_s:
                return format_output(format, [1, 1])
            elif event.key == pygame.K_d:
                return format_output(format, [1, 2])
            elif event.key == pygame.K_a:
                return format_output(format, [1, 3])
    
    return None
