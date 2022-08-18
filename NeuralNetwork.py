import random
from math import exp


# ReLU activation function: 0 if x > 0 else 0
def ReLU(x) -> list[float]:
    return [max(i, 0) for i in x]

# map the values to probabilities such that total probability = 1
def softmax(x: list[int | float]) -> list[float]:
    denominator = sum([exp(i) for i in x])
    return [exp(i)/denominator for i in x]

# cost function: error between predicted and actual value
def cost_function(node_output: int | float, expected_output: int | float) -> float:
    error = node_output - expected_output
    return error * error

# Custom error for no inputs given (Rust like programming)
class NoInputsGivenError(Exception):
    pass


# Object that holds the data and it's properties
class DataPoint:
    """
    Inputs: list of all the input values (x)
    Outputs: list of the expected output (y) => {
        Based on the label, the values at the label's indices are
        set to 1, and the rest are set to 0. So, if the label for 
        a specific input is '2' where the possible labels are,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], then the param 'ouptuts' should be,
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
    }
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

# A single Node object in the neural network:   Node / Neuron
class Node:
    """
    Takes a set of inputs and makes weights and biases based on it.
    """
    def __init__(self, inputs: list[int | float]):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.bias = random.uniform(-1, 1)

    def output(self):
        output_value = 0

        for i, w in zip(self.inputs, self.weights):
            # weighted input
            output_value += i * w

        output_value += self.bias
        return output_value

    def __int__(self):
        return self.output()

# A layer consists of many nodes, receives data from previous layer and
# feeds to the next layer
class Layer:
    def __init__(self, number_of_nodes, input_layer):
        if input_layer != []:
            self.nodes = [Node(input_layer) for _ in range(number_of_nodes)]
        else:
            raise NoInputsGivenError("Declared as non input layer, but no inputs were given")

    # calculates the output for each 
    def condense(self, activation_function):
        self.outputs = activation_function([node.output() for node in self.nodes])
        return self.outputs

    #  This method is ONLY for an output layer
    def cost_function(self, expected_outputs):
        cost = 0
        for i in range(len(self.nodes)):
            cost += cost_function(self.outputs[i], expected_outputs[i])
        
        return cost

    def __len__(self):
        return len(self.nodes)

class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        self.input_nodes  = input_nodes 
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes 

    def feed_forward(self, input_layer: list[float | int]):
        self.input_layer = input_layer[:self.input_nodes]
        self.hidden_layer = Layer(self.hidden_nodes, self.input_layer)
        self.output_layer = Layer(self.output_nodes, self.hidden_layer.condense(ReLU))
        self.output = self.output_layer.condense(softmax)
        return self.output

    def cost(self, data_points: DataPoint):
        total_cost = 0
        for data_point in data_points:
            self.feed_forward(data_point.inputs)
            total_cost += self.output_layer.cost_function(data_point.outputs)

        return total_cost / len(data_points)
