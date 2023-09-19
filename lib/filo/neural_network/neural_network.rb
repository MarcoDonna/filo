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

        def initialize layers: nil, optimizer: nil
            raise ArgumentError.new if layers.nil?
            @layers = layers
            @optimizer = optimizer
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
        def train x: nil, y: nil, epochs: nil, target_loss: nil, batch_size: nil, batches: nil, optimizer: nil, debug: false
            raise ArgumentError.new if x.nil? or y.nil?

            @optimizer = optimizer unless optimizer.nil?

            batch_size = x.row_size if batch_size.nil?
            batch_size = (x.row_size / batches.to_f).ceil unless batches.nil?

            input_batches = x.to_a.each_slice(batch_size).map { |mat| mat.to_matrix }
            target_batches = y.to_a.each_slice(batch_size).map { |mat| mat.to_matrix }

            training_pairs = input_batches.zip(target_batches)

            epochs.times do |e|
                p e if debug === true
                training_pairs.each do |input, target|
                    forward(input)
                    backprop(target)
                    optimize()
                end

                # Check for average output layer target loss exit condition
                unless target_loss.nil?
                    avg_loss_at_timestep = output_layer.loss_metric.last.inject(0) { |acc, val| acc + val} / output_layer.size
                    return if avg_loss_at_timestep <= target_loss
                end
            end
        end

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
            if(@optimizer.nil?)
                @layers.each { |layer| layer.optimize if layer.respond_to?(:optimize) }
            else
                biases = @optimizer.optimize_biases(biases: flattened_biases(), gradients: flattened_biases_gradients())
                weights = @optimizer.optimize_weights(weights: flattened_weights(), gradients: flattened_weights_gradients())
                distribute_biases(biases)
                distribute_weights(weights)
            end
        end

        private

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

        # Returns a list of flattened weights of all the layers.
        #
        def flattened_weights
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights) ? layer.weights.to_a : []) }.flatten]
        end

        # Returns a list of flattened weights gradients of all the layers.
        #
        def flattened_weights_gradients
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights_gradient) ? layer.weights_gradient.to_a : []) }.flatten]
        end

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
