import pygame

def get_direction():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return [0, 0]
            elif event.key == pygame.K_DOWN:
                return [0, 1]
            elif event.key == pygame.K_RIGHT:
                return [0, 2]
            elif event.key == pygame.K_LEFT:
                return [0, 3]
            elif event.key == pygame.K_w:
                return [1, 0]
            elif event.key == pygame.K_s:
                return [1, 1]
            elif event.key == pygame.K_d:
                return [1, 2]
            elif event.key == pygame.K_a:
                return [1, 3]
    return None