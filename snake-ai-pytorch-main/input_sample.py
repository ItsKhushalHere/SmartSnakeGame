import pygame
from pygame.locals import *

pygame.init()

# Set up the display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Text Input Event")

# Variables for text input
input_text = ""
font = pygame.font.Font(None, 32)
text_color = pygame.Color('black')

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_BACKSPACE:
                # Remove the last character if backspace is pressed
                input_text = input_text[:-1]
            elif event.key == K_RETURN:
                # Check if the input text matches the desired text
                if input_text == "hello":
                    print("Hello typed!")  # Replace with your desired action
                else:
                    print("Text doesn't match!")
                input_text = ""
            else:
                # Add the pressed character to the input text
                input_text += event.unicode

    screen.fill((255, 255, 255))  # Clear the screen
    # Render the input text
    text_surface = font.render(input_text, True, text_color)
    screen.blit(text_surface, (10, 10))
    pygame.display.flip()

pygame.quit()
