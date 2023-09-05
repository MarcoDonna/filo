#Copyright 2023 Marco Donna
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

module Filo
    class NeuralNetwork

        def initialize config
            @config = config
            @layers = @config[:layers] || []
            @depth = @layers.size
        end

        def input_layer
            @layers.first
        end

        def output_layer
            @layers.last
        end

        def output
            output_layer.output
        end

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

        def forward input
            input_layer.forward(input)
            # Each layer except input layer
            (@depth - 1).times do |layer_index|
                layer_index = layer_index + 1
                @layers[layer_index].forward(@layers[layer_index - 1])
            end
            return output
        end

        def backprop target
            output_layer.backprop(target)
            # Each layer except input and output layer, starting from last
            (@depth - 2).times do |layer_index|
                layer_index = @depth - layer_index - 2
                @layers[layer_index].backprop(@layers[layer_index + 1])
            end
        end

        # Optimize the weights and biases in the network
        def optimize
            if(@config[:optimizer].nil?)
                @layers.each { |layer| layer.optimize if layer.respond_to?(:optimize) }
            else
                biases = @config[:optimizer].optimize_biases(flattened_biases(), flattened_biases_gradients())
                weights = @config[:optimizer].optimize_vector(flattened_weights(), flattened_weights_gradients())
                distribute_biases(biases)
                distribute_weights(weights)
            end
        end

        private

        def flattened_biases
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:biases) ? layer.biases.to_a : []) }.flatten]
        end

        def flattened_biases_gradients
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:biases_gradient) ? layer.biases_gradient.to_a : []) }.flatten]
        end

        def distribute_biases biases
            biases = biases.to_a
            @layers.each do |layer|
                if(layer.respond_to?(:biases))
                    layer.biases = Vector[*biases[0, layer.size]]
                    biases = biases.drop(layer.size)
                end
            end
        end

        def flattened_weights
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights) ? layer.weights.to_a : []) }.flatten]
        end

        def flattened_weights_gradients
            Vector[*@layers.inject([]) { |list, layer| list << (layer.respond_to?(:weights_gradient) ? layer.weights_gradient.to_a : []) }.flatten]
        end

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
