
import pygame
pygame.font.init()
import os


D_WIDTH = 500
D_HEIGHT = 800
display = pygame.display.set_mode((D_WIDTH, D_HEIGHT))

# Assets
BIRD_IMG =  pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bird.png'))).convert_alpha()
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'pipe.png'))).convert_alpha()
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'base.png'))).convert_alpha()
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bg - dark.png'))).convert_alpha()
ARCADE_FONT = pygame.font.Font(os.path.join('assets', 'ARCADECLASSIC.TTF'), 24)

# Algorithm configurations
POPULATION_SIZE = 15                # Size of the population
GENE_LENGTH = 4                     # length of the gene information
MUTATION_RATE = 0.02                # rate of mutation chances
MUTATION_RANGE_LOW = -0.4           # Lowest possible mutation weight value
MUTATION_RANGE_HIGH = 0.4           # highest possible mutation weight value
AGGRESSIVE_MUTATION_CHANCE = 0.02   # chances of aggressively mutating a genome


pygame.display.set_caption('Flappy bird')
pygame.display.set_icon(BIRD_IMG)

def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect)
