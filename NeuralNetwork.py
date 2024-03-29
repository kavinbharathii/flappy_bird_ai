import random
import numpy as np
from activation_funcs import sigmoid, tanh
from settings import MUTATION_RANGE_LOW, MUTATION_RANGE_HIGH

class NotImplementedError(Exception):
    pass

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def feed_forward(self, inputs):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def feed_forward(self, input_data):
        input_data = np.array(input_data)
        return np.dot(input_data, self.weights) + self.bias


class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def feed_forward(self, input_data):
        return self.activation_function(input_data)

class Network:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.feed_forward(output)

        return output


def intertwine(father, mother):
    child = Network()

    for layer in range(len(mother.layers)):
        # If it is a FCLayer, then mutate it
        if type(mother.layers[layer]) == FCLayer:
            out = FCLayer(len(mother.layers[layer].weights), len(mother.layers[layer].weights[0]))
            for num_nodes in range(len(mother.layers[layer].weights)):
                if num_nodes % 2 == 0:
                    out.weights[num_nodes] = father.layers[layer].weights[num_nodes]
                else:
                    out.weights[num_nodes] = mother.layers[layer].weights[num_nodes]

            if random.uniform(-1, 1) > 0:
                out.bias = father.layers[layer].bias
            else:
                out.bias = mother.layers[layer].bias 

        # If not, then it is a activation layer, so add it as it is.
        else:
            out = father.layers[layer]

        child.add(out)

    return child


# Slighlty mutate the weights with smalls tweaks

# new weights = old weights + (small range of values)
def evolve(genome: Network):
    evolved_genome = Network()
    
    for layer_index in range(len(genome.layers)):

        # If it's a FCLayer
        if type(genome_layer := genome.layers[layer_index]) == FCLayer:

            number_of_weight_rows = len(genome_layer.weights)
            number_of_weight_cols = len(genome_layer.weights[0])
            weight_tweaks = np.random.uniform(
                size = (number_of_weight_rows, number_of_weight_cols),
                low = MUTATION_RANGE_LOW,
                high = MUTATION_RANGE_HIGH)

            # initializing a placeholder FCLayer
            evolved_layer = FCLayer(1, 1)
            evolved_layer.weights = genome_layer.weights + weight_tweaks
            evolved_genome.add(evolved_layer)

        # If it's an activation layer
        else:
            evolved_genome.add(genome.layers[layer_index])

    return evolved_genome


# aggressive mutation completely changes the weights of the layer
# it produces more variety in the population

def aggressive_mutation(genome: Network):
    evolved_genome = Network()
    
    for layer_index in range(len(genome.layers)):

        # If it's a FCLayer
        if type(genome_layer := genome.layers[layer_index]) == FCLayer:

            number_of_weight_rows = len(genome_layer.weights)
            number_of_weight_cols = len(genome_layer.weights[0])
            
            new_weights = np.random.rand(number_of_weight_rows, number_of_weight_cols) - 0.5
            
            # initializing a placeholder FCLayer
            evolved_layer = FCLayer(1, 1)
            evolved_layer.weights = new_weights
            evolved_genome.add(evolved_layer)

        # If it's an activation layer
        else:
            evolved_genome.add(genome.layers[layer_index])

    return evolved_genome

