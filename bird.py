import pygame
import numpy as np
import random
from settings import *
from NeuralNetwork import FCLayer, Network, ActivationLayer, intertwine, evolve, aggressive_mutation
from activation_funcs import sigmoid, tanh


def mapn(value, act_lower, act_upper, to_lower, to_upper):
    return to_lower + (to_upper - to_lower) * ((value - act_lower) / (act_upper - act_lower))


class Bird:
    def __init__(self, x, y, vel, img):
        self.x = x
        self.y = y
        self.vel = vel
        self.img = img
        self.acc = 0.005
        self.tick_count = 0
        self.jump_vel = -3
        self.flapping = False
        self.terminal_vel = 7
        self.height = self.y
        self.tilt = 0
        self.max_rotation = 25
        self.rotation_velocity = 3
        self.fitness = -1
        self.dead = False
        self.generation = -1
        self.age = 0
        self.visual_inputs = []
        self.INPUT_SIZE = 4
        self.HIDDEN_LAYER = 5
        self.OUTPUT_SIZE = 2

    def draw(self, win):
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def flap(self):
        if not self.flapping:
            self.vel = self.jump_vel
            self.tick_count = 0
            self.flapping = True

    def move(self):
        self.tick_count += 1

        # for downward acceleration
        # v = u + at
        # we are updating the velocity value and we increment the 
        # bird's y position w.r.t the velocity.
        self.vel = self.vel + self.acc * self.tick_count

        # terminal velocity
        if self.vel >= self.terminal_vel:
            # self.vel = (self.vel/abs(self.vel)) * self.terminal_vel
            self.vel = self.terminal_vel

        # changing the bird's position w.r.t velocity
        self.y += self.vel

        # checking if the bird is flapping by checking the bird's velocity
        if self.vel > 0:
            self.flapping = False

        # tilt up
        if self.vel < 0:  
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        # tilt down
        else:  
            if self.tilt > -self.max_rotation:
                self.tilt -= self.rotation_velocity
        
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

    def create_genes(self):
        self.nn = Network()
        self.nn.add(FCLayer(self.INPUT_SIZE, self.HIDDEN_LAYER))
        self.nn.add(ActivationLayer(sigmoid))
        self.nn.add(FCLayer(self.HIDDEN_LAYER, self.OUTPUT_SIZE))
        self.nn.add(ActivationLayer(tanh))

    def calc_fitness(self):
        # using a fitness function to find the fitness of
        # the specific genome, and use it as the metric to
        # improve it's probability of becoming a parent
        return self.age / 100

    def crossover(self, partner):
        child = Bird(30, D_HEIGHT // 2, 0, BIRD_IMG)
        child.nn = intertwine(self.nn, partner.nn)
        return child 

    def evolve_genome(self):
        child = Bird(30, D_HEIGHT // 2, 0, BIRD_IMG)
        chance_of_aggressive_mutation = random.random()

        if chance_of_aggressive_mutation <= AGGRESSIVE_MUTATION_CHANCE:
            child.nn = aggressive_mutation(self.nn)
            print("Aggressive mutation")
        else:
            child.nn = evolve(self.nn)

        return child

    def look(self, pipe):
        self.visual_inputs = []

        # horizontal distance from pipe
        horizontal_distance = pipe.x - self.x
        mapped_hd = mapn(horizontal_distance, 0, D_WIDTH, 0, 1)
        
        # bird's velocity
        velocity = self.vel / self.terminal_vel
    
        # vertical distance from top pipe
        top_pipe_height = pipe.height - GAP_HEIGHT
        mapped_tph = mapn(top_pipe_height, 0, D_HEIGHT - BASE_HEIGHT - GAP_HEIGHT, 0, 1)
        
        # vertical distance from bottom pipe
        bottom_pipe_height = pipe.height
        mapped_bph = mapn(bottom_pipe_height, GAP_HEIGHT, D_HEIGHT - BASE_HEIGHT, 0, 1)

        self.visual_inputs.append(mapped_hd)
        self.visual_inputs.append(velocity)
        self.visual_inputs.append(mapped_tph)
        self.visual_inputs.append(mapped_bph)

    def debug(self, win, pipe):
        pygame.draw.line(win,  (191, 97, 106), (self.x, self.y), (pipe.x, self.y), 5)
        pygame.draw.line(win, (163, 190, 140), (self.x, self.y), (pipe.x, pipe.height - 200), 5)
        pygame.draw.line(win, (129, 161, 193), (self.x, self.y), (pipe.x, pipe.height), 5)

    def think(self):
        self.visual_inputs = np.array(self.visual_inputs)
        decision = self.nn.predict([self.visual_inputs])
        
        if decision[0][0] > decision[0][1]:
            return 0
        else:
            return 1