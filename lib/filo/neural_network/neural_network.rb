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
            options[:epochs].times do
                forward(input)
                backprop(target)
                optimize()
            end
        end

        def forward input
            output = input_layer.forward(input)
            # Each layer except input layer
            (@depth - 1).times do |layer_index|
                layer_index = layer_index + 1
                @layers[layer_index].forward(@layers[layer_index - 1])
            end
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
            @layers.each do |layer|
                layer.optimize if layer.respond_to?(:optimize)
            end
        end
    end
end
