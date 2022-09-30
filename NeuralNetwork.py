from random import uniform as r_uniform
import numpy as np
from activation_funcs import sigmoid, tanh

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

            if r_uniform(-1, 1) > 0:
                out.bias = father.layers[layer].bias
            else:
                out.bias = mother.layers[layer].bias 

        # If not, then it is a activation layer, so add it as it is.
        else:
            out = father.layers[layer]

        child.add(out)

    return child


# # [T E S T I N G]
# father = Network()
# father.add(FCLayer(2, 3))
# father.add(ActivationLayer(tanh))
# father.add(FCLayer(3, 1))
# father.add(ActivationLayer(tanh))
# for l in father.layers:
#     if type(l) == FCLayer: 
#         print(l.weights)

# print()

# mother = Network()
# mother.add(FCLayer(2, 3))
# mother.add(ActivationLayer(tanh))
# mother.add(FCLayer(3, 1))
# mother.add(ActivationLayer(tanh))
# for l in mother.layers:
#     if type(l) == FCLayer: 
#         print(l.weights)

# print()

# child = intertwine(father, mother)
# for l in child.layers:
#     if type(l) == FCLayer: 
#         print(l.weights)

# print()

# print("STATS")
# for i in range(len(child.layers)):
#     if type(child.layers[i]) == FCLayer:
#         print(f"Layer {i}")
#         print(child.layers[i] == father.layers[i])
#         print(child.layers[i] == mother.layers[i])
#         print()


# w = np.array([[1, 0]])
# print(father.predict(w))
# print(mother.predict(w))
# print(child.predict(w))