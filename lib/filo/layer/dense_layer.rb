module Filo
    module Layer

        def self.DenseLayer *args
            if args.size == 1 and args[0].is_a?(Hash)
                DenseLayer.new(args[0])
            elsif args.size == 3 and args[2].is_a?(Hash)
                DenseLayer(args[2].merge(input_size: args[0], size: args[1]))
            else
                raise InvalidArgumentError.new
            end
        end

        class DenseLayer < Layer
            attr_reader :biases, :weights
            attr_reader :input, :before_activation, :output, :error

            def initialize config={}
                super(config)
                raise InvalidArgumentError.new("No :activation_function in layer config") if @config[:activation_function].nil?

                @biases = Vector.zero(@size).map { rand_weight() }
                @weights = Matrix.build(@size, @input_size) { rand_weight() }
            end

            def biases= biases
                raise ShapeError.new unless biases.size == @size
                @biases = biases
            end

            def weights= weights
                raise ShapeError.new unless weights.column_size == @input_size and weights.row_size == @size
                @weights = weights
            end

            def forward prev_layer
                @input = prev_layer.output
                @before_activation = (@input * @weights.t).add_vector(@biases)
                @output = @config[:activation_function].apply_activation_function(@before_activation)
            end

            def backprop next_layer
                @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(next_layer.weighted_error)
            end

            def optimize
                @biases = @config[:optimizer].optimize_weights(@biases, biases_gradient)
                @weights = @config[:optimizer].optimize_weights(@weights, weights_gradient)
            end

            def weighted_error
                @error * @weights
            end

            def weights_gradient
                @error.t * @input / @output.row_size
            end

            def biases_gradient
                #Sum error over batch, divide by @output.row_size to get average over batch
                Vector[*@error.t.map_row { |row| row.reduce(0) { |acc, val| acc + val } }] / @output.row_size
            end

            private

            def rand_weight
                rand()
            end
        end

    end
end
