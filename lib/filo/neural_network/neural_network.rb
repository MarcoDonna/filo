#--
# Copyright 2023 Marco Donna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#++

module Filo
    class NeuralNetwork

        # call-seq:
        # new(layers: Array, optimizer: Optimizer=nil) -> Filo::NeuralNetwork
        #
        def initialize config
            @config = config
            @layers = @config[:layers] || []
            @depth = @layers.size
        end

        def input_layer # :nodoc:
            @layers.first
        end

        def output_layer # :nodoc:
            @layers.last
        end

        # Returns the output of the output layer or nil if forward methods was not called before.
        #
        def output
            output_layer.output
        end

        # call-seq:
        # train(Matrix, Matrix, {batch_size: Numeric, epochs: Numeric, target_loss: Numeric=nil})
        # train(Matrix, Matrix, {batches: Numeric, epochs: Numeric, target_loss: Numeric=nil})
        #
        # If no batch_size or batches is passed, batch_size is set to take entire dataset.
        # If only batches is passed, batches is used to compute batch_size
        #
        # Fit the model to the given data by minimizing the error.
        #
        #   # Create network
        #   network = Filo::NeuralNetwork.new(layers: layers, optimizer: Filo::Optimizer.SGD(learning_rate: 0.1))
        #
        #   # Train network
        #   network.train(data_features, data_targets, epochs: 50000, batches: 1, target_loss: 0.05)
        #
        def train input, target, options
            # Split data in  batches if batch_size or batches is set
            options[:batch_size] = input.row_size if options[:batch_size].nil?
            options[:batch_size] = (input.row_size / options[:batches].to_f) unless options[:batches].nil?
            input_batches = input.to_a.each_slice(options[:batch_size]).map { |mat| Matrix[*mat] }
            target_batches = target.to_a.each_slice(options[:batch_size]).map { |mat| Matrix[*mat] }

            training_pairs = input_batches.zip(target_batches)

            options[:epochs].times do
                training_pairs.each do |input, target|
                    forward(input)
                    backprop(target)
                    optimize()
                end
                # Check for average output layer target loss exit condition
                unless options[:target_loss].nil?
                    avg_loss_at_timestep = output_layer.loss_metric.last.inject(0) { |acc, val| acc + val} / output_layer.size
                    return if avg_loss_at_timestep <= options[:target_loss]
                end
            end
        end

        # call-seq:
        # forward(Matrix) -> Matrix
        #
        # Takes a Matrix of inputs and propagates them trough the network. Returns the output of the output layer.
        #
        #   # predict something
        #   network.forward(Matrix[[0, 0], [0, 0]])
        #   # => Matrix, same number of rows, one column for each neuron in output layer.
        #
        def forward input
            input_layer.forward(input)
            # Each layer except input layer
            (@depth - 1).times do |layer_index|
                layer_index = layer_index + 1
                @layers[layer_index].forward(@layers[layer_index - 1])
            end
            return output
        end

        # call-seq:
        # backprop(Matrix) -> Matrix
        #
        # Takes a Matrix of targets, computes the loss and backprops it trought the network.
        #
        def backprop target
            output_layer.backprop(target)
            # Each layer except input and output layer, starting from last
            (@depth - 2).times do |layer_index|
                layer_index = @depth - layer_index - 2
                @layers[layer_index].backprop(@layers[layer_index + 1])
            end
        end

        # Tries to optimize the weights of each layer.
        #
        # If a optimizer is specified in the config passed when building the network,
        # flattens the biases and weights and optimizes them in a single pass.
        #
        # Otherwise tries to call the method +optimize+ of each layer.
        #
        def optimize
            if(@config[:optimizer].nil?)
                @layers.each { |layer| layer.optimize if layer.respond_to?(:optimize) }
            else
                biases = @config[:optimizer].optimize_biases(flattened_biases(), flattened_biases_gradients())
                weights = @config[:optimizer].optimize_weights(flattened_weights(), flattened_weights_gradients())
                distribute_biases(biases)
                distribute_weights(weights)
            end
        end

        private

        # call-seq:
        # flattened_biases -> Vector
        #
        # Returns a list of flattened biases of all the layers.
        #
        def flattened_biases
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:biases) ? layer.biases.to_a : []) }.flatten]
        end

        # call-seq:
        # flattened_biases_gradients -> Vector
        #
        # Returns a list of flattened biases gradients of all the layers.
        #
        def flattened_biases_gradients
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:biases_gradient) ? layer.biases_gradient.to_a : []) }.flatten]
        end

        # call-seq:
        # distribute_biases(Vector)
        #
        # Given a list of Vectors, distributes the to the layers following the order:
        # 1. layers from input_layer to output_layer.
        # 2. neurons from 0 to layer_size.
        #
        def distribute_biases biases
            biases = biases.to_a
            @layers.each do |layer|
                if(layer.respond_to?(:biases))
                    layer.biases = Vector[*biases[0, layer.size]]
                    biases = biases.drop(layer.size)
                end
            end
        end

        # call-seq:
        # flattened_weights -> Vector
        #
        # Returns a list of flattened weights of all the layers.
        #
        def flattened_weights
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights) ? layer.weights.to_a : []) }.flatten]
        end

        # call-seq:
        # flattened_weights -> Vector
        #
        # Returns a list of flattened weights gradients of all the layers.
        #
        def flattened_weights_gradients
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights_gradient) ? layer.weights_gradient.to_a : []) }.flatten]
        end

        # call-seq:
        # distribute_biases(Vector)
        #
        # Given a list of Vectors, distributes the to the layers following the order:
        # 1. layers from input_layer to output_layer.
        # 2. neurons from 0 to layer_size.
        # 3. weights from 0 to input_size.
        #
        def distribute_weights weights
            weights = weights.to_a
            @layers.each do |layer|
                if(layer.respond_to?(:weights))
                    # Reshape weights to matrix [nurons x weights]
                    current_layer_weights = []
                    layer.size.times do
                        current_layer_weights << weights[0, layer.input_size]
                        weights = weights.drop(layer.input_size)
                    end
                    layer.weights = Matrix[*current_layer_weights]
                end
            end
        end
    end
end
