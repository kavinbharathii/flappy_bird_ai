
# Introduction:

This project includes [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) and Evolving Neural Networks, in-order to create a optimal player in the context of a [Flappy Bird Game](https://en.wikipedia.org/wiki/Flappy_Bird). The usage of both the Genetic Algorithm and the evolving neural networks gives us an edge over other algorithms. 

# Working principle:

This works by the principle of natural selection. selective breeding of neural networks and evolution of genomes. Various genomes with different traits and characteristics are exposed to the Flappy Bird game environment where they have to flay over `Pipes` in-order to survive, and to get a high score.

Given two genomes, with the following gene sets

![](https://res.cloudinary.com/practicaldev/image/fetch/s--A6H_pBdv--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/vdbrhqofvcyiasuijjth.png)


the offspring of the genomes can be a cross-over between the genomes, resulting in the following gene set.

![](https://res.cloudinary.com/practicaldev/image/fetch/s--bMf3gTCi--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/fj74k8fjf1098mv2pe0a.png)

This environment can be controlled with various settings, found in the `settings.py` file

```python
# Algorithm configurations
POPULATION_SIZE = 10                # Size of the population
GENE_LENGTH = 4                     # length of the gene information
MUTATION_RATE = 0.02                # rate of mutation chances
MUTATION_RANGE_LOW = -0.4           # Lowest possible mutation weight value
MUTATION_RANGE_HIGH = 0.4           # highest possible mutation weight value
AGGRESSIVE_MUTATION_CHANCE = 0.02   # chances of aggressively mutating a genome
```

This gives us control over the environment and the learning process. The various genomes are neural networks with evolution mechanics built into them. They evolve using the following code logic.

```python
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
```

found in the `NueralNetwork.py` file. They can also be *aggressively mutated* which gives us varied results,

```python
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
```

The chances of aggressive mutation is 2% which can be changed in the settings. A bird (genome) is a neural network with the following architecture,

```python
def create_genes(self):
	self.nn = Network()
	self.nn.add(FCLayer(self.INPUT_SIZE, self.HIDDEN_LAYER))
	self.nn.add(ActivationLayer(sigmoid))
	self.nn.add(FCLayer(self.HIDDEN_LAYER, self.OUTPUT_SIZE))
	self.nn.add(ActivationLayer(tanh))
```

and the **fitness** of each genome is calculated by the "age" of the bird:

```python
def calc_fitness(self):
	# using a fitness function to find the fitness of
	# the specific genome, and use it as the metric to
	# improve it's probability of becoming a parent
	return self.age / 100
```

Each genome has four sensory inputs, the horizontal location, the velocity of the bird, the top height of the next pipe, the bottom height of the next pipe.

```python
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
```

By selecting the best bunch from each episode and breeding + evolving them, a optimal player converges.

# References:

Genetic Algorithm Wiki: https://en.wikipedia.org/wiki/Genetic_algorithm

Neural Network Wiki: https://en.wikipedia.org/wiki/Neural_network

Kavin's article on Genetic Algorithm: https://dev.to/kavinbharathi/genetic-algorithm-in-action-3ilj

GitHub Repo for the Project: https://github.com/kavinbharathii/flappy_bird_ai

