import pygame

def get_direction():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return 0
            elif event.key == pygame.K_DOWN:
                return 1
            elif event.key == pygame.K_RIGHT:
                return 2
            elif event.key == pygame.K_LEFT:
                return 3
    return None