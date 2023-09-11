require 'filo'

# Defining training data features
xor_f = Matrix[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Defining traing data classes (XOR problem as one-hot vector)
xor_t = Matrix[
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
]

# Creating the layers of the Neural Network
# This Network is made of
# - InputLayer with 2 Nodes, pne for each column in the training features
# - DenseLayer with 3 Nodes, each Node has 2 weights (one for each node in the InputLayer).
#   - This layer uses the Sigmoid activation function
# - OutputLayer with 2 Nodes, one for each column in the training targets. Each node has 3 weights, one for each node in the DenseLayer.
#   - The OutputLayer uses the Sigmoid Activation Function and MSE (MeanSquareError) as loss function.
layers =[
    Filo::Layer.InputLayer(2),
    Filo::Layer.DenseLayer(2, 3, activation_function: Filo::Activation.Sigmoid),
    Filo::Layer.OutputLayer(3, 2, activation_function: Filo::Activation.Softmax, loss_function: Filo::Loss.MSE)
]

# Creating the Neural Network, use Stochastic Gradient Descent as Optimizer
network = Filo::NeuralNetwork.new(layers: layers, optimizer: Filo::Optimizer.SGD(learning_rate: 0.2))

# Train the Neural Network to solve the XOR problem
# Target loss is likely to be reached before running out of epochs
network.train(xor_f, xor_t, epochs: 50000, batches: 4, target_loss: 0.01)

# Log the loss metric at the start and at the end of training
p network.output_layer.loss_metric.first
p network.output_layer.loss_metric.last

# Forward the inputs of the XOR problem to the Neural Network and print the output
# The output can also be accessed using the method +output+
p network.forward(xor_f)
