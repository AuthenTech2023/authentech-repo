import pygame
import sys
import random
import numpy as np
from pygame.locals import *
import time

sys.stdout = open("Subject0.txt", 'w')

### SETTINGS ###
random.seed(69)             # SAME TEST FOR EVERYONE?
width = 50                  # Square width
height = 50                 # Square height
SCREEN_WIDTH = 800          # Program screen width
SCREEN_HEIGHT = 600         # Program screen height
MIN_WIDTH_CHANGE = 120      # Min width distance between old and new square
MAX_WIDTH_CHANGE = 210      # Max width distance between old and new square
MIN_HEIGHT_CHANGE = 100     # Min height distance between old and new square
MAX_HEIGHT_CHANGE = 160     # Max width distance between old and new square
### /SETTINGS ###

### Variable Initialization ###
wrong = 0
running = True
flag = True
### Variable Initialization ###



### Pre-program execution ###
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
intro_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font = pygame.font.SysFont('Arial', 25)
text = font.render('Click \'START\' to begin', True, (255, 255, 255), (0, 0, 0))
text2 = font.render('START', True, (255, 255, 255), (0, 0, 0))
textRect2 = text2.get_rect()
textRect = text.get_rect()
textRect.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT//4)
textRect2.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT//1.5)
### Pre-program execution ###

while running:                                                                                                          # Pygame execution loop
    ### Create start screen and wait ###
    screen.fill((0, 0, 0))
    screen.blit(text, textRect)
    screen.blit(text2, textRect2)
    pygame.display.update()
    event = pygame.event.wait()
    ### /Create start screen and wait ###
    if event.type == pygame.QUIT:
        running = False
    elif event.type == pygame.MOUSEBUTTONDOWN and textRect2.left < event.pos[0] < textRect2.right \
          and textRect2.top < event.pos[1] < textRect2.bottom and pygame.mouse.get_pressed()[0]:
        ### Draw initial square ###
        det_button = random.randint(0, 2)
        rect_x_origin = random.randint(0, SCREEN_WIDTH - width)
        rect_y_origin = random.randint(height, SCREEN_HEIGHT - height)
        screen.fill((0, 0, 0))
        square = pygame.draw.rect(screen, (0, 0, 255), ((rect_x_origin, rect_y_origin), (width, height)))
        if det_button == 0:
            screen.blit(font.render('L', True, (255, 255, 255)),
                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
        elif det_button == 1:
            screen.blit(font.render('M', True, (255, 255, 255)),
                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
        elif det_button == 2:
            screen.blit(font.render('R', True, (255, 255, 255)),
                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
        # pygame.draw.circle(surface=screen, color=(0, 0, 0), center=(square.centerx, square.centery), radius=3, width=3) # Click as close to the center as possible?
        pygame.display.update()
        start = time.time()
        ### /Draw initial square ###
        while flag:
            pygame.display.update()
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                running = False
                flag = False
            if event.type == pygame.MOUSEMOTION:
                print(event.pos, -1, -1, -1, (-1, -1), ';')
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if square.bottom > event.pos[1] > square.top and square.right > event.pos[0] > square.left:             # If user clicked in the square
                    if (pygame.mouse.get_pressed()[0] and det_button == 0) \
                            or (pygame.mouse.get_pressed()[1] and det_button == 1) \
                            or (pygame.mouse.get_pressed()[2] and det_button == 2):                                     # If the user clicked with the correct button
                        end = time.time()
                        print("{} {} {}".format(event.pos, np.argmax(pygame.mouse.get_pressed()), wrong), end=" ")
                        print("%.6f" % (end - start), end=" ")
                        center = (abs(square.centerx - event.pos[0]), abs(square.centery - event.pos[1]))
                        print(center, ';')
                        wrong = 0
                        screen.fill((0, 0, 0))
                        ### Increase randomization of square spawn location ###                                         # TODO: Formal algorithm to implement better randomization?
                        if rect_x_origin == SCREEN_WIDTH - width:                                                       # If square was all the way on the right side, spawn somewhere near the left side
                            rect_x_origin = random.randint(SCREEN_WIDTH // 6, SCREEN_WIDTH // 4)
                        elif rect_x_origin == 0:
                            rect_x_origin = random.randint(SCREEN_WIDTH //3, SCREEN_WIDTH-width)
                        if rect_y_origin == SCREEN_HEIGHT - height:                                                     # If square was all the way on the bottom, spawn somewhere near the top
                            rect_y_origin = random.randint(SCREEN_HEIGHT // 6, SCREEN_WIDTH // 4)
                        elif rect_y_origin == 0:
                            rect_y_origin = random.randint(SCREEN_HEIGHT // 3, SCREEN_HEIGHT - height)
                        else:
                            random_sign = random.randint(0, 1)
                            if random_sign == 0:
                                rect_x_origin = min(rect_x_origin + random.randint(MIN_WIDTH_CHANGE, MAX_WIDTH_CHANGE),
                                                    SCREEN_WIDTH - width)
                                rect_y_origin = min(
                                    rect_y_origin + random.randint(MIN_HEIGHT_CHANGE, MAX_HEIGHT_CHANGE),
                                    SCREEN_HEIGHT - height)
                            else:
                                rect_x_origin = max(rect_x_origin - random.randint(MIN_WIDTH_CHANGE, MAX_WIDTH_CHANGE),
                                                    0)
                                rect_y_origin = max(
                                    rect_y_origin - random.randint(MIN_HEIGHT_CHANGE, MAX_HEIGHT_CHANGE), 0)
                        ### /Increase randomization of square spawn location ###
                        square = pygame.draw.rect(screen, (0, 0, 255), ((rect_x_origin, rect_y_origin), (width, height)))
                        ### Determine and draw next square's required mouse button ###
                        det_button = random.randint(0, 2)
                        if det_button == 0:
                            screen.blit(font.render('L', True, (255, 255, 255)),
                                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
                        elif det_button == 1:
                            screen.blit(font.render('M', True, (255, 255, 255)),
                                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
                        elif det_button == 2:
                            screen.blit(font.render('R', True, (255, 255, 255)),
                                        (rect_x_origin + (width / 2.5), rect_y_origin + height / 5))
                        ### /Determine and draw next square's required mouse button ###
                        start = time.time()
                    else:
                        wrong += 1
                else:
                    wrong += 1                                                                                          # TODO: Penalizes user for click outside box. Subject to change?
sys.stdout.close()
