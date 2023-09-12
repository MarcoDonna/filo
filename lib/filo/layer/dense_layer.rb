module Filo
    module Layer

        # call-seq:
        # DenseLayer(input_size: Numeric, size: Numeric, activation_function: Activation, optimizer: Optimzer=nil) -> Filo::Layer::DenseLayer
        # DenseLayer(Numeric input_size, Numeric size, config={}) -> Filo::Layer::DenseLayer
        #
        #   dense_layer = Filo::Layer.DenseLayer(2, 4, activation_function: Activation::Sigmoid.new)
        #   dense_layer = Filo::Layer.DenseLayer(input_size: 2, size: 4, activation_function: Activation::Sigmoid.new)
        #
        # Returns a new Intance of DenseLayer.
        #
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
            attr_reader :biases, :weights # :nodoc:
            attr_reader :input, :before_activation, :output, :error # :nodoc:

            def initialize config={} # :notnew:
                super(config)
                raise InvalidArgumentError.new("No :activation_function in layer config") if @config[:activation_function].nil?

                @biases = Vector.zero(@size).map { rand_weight() }
                @weights = Matrix.build(@size, @input_size) { rand_weight() }
            end

            # call-seq:
            # biases=(Vector)
            #
            # Set the layer biases to be the Vector passed as argument.
            #
            def biases= biases
                raise ShapeError.new unless biases.size == @size
                @biases = biases
            end

            # call-seq:
            # weights=(Matrix)
            #
            # Set the layer weights to be the Matrix passed as arguments.
            #
            def weights= weights
                raise ShapeError.new unless weights.column_size == @input_size and weights.row_size == @size
                @weights = weights
            end

            # call-seq:
            # forward(Layer)
            #
            # Takes the output of the layer passed as argument and perfmorms forward pass, calculating @input, @before_activation and @output.
            #
            # Forward pass consists (simplified) of biases + weights * inputs.
            #
            def forward prev_layer
                @input = prev_layer.output
                @before_activation = (@input * @weights.t).add_vector(@biases)
                @output = @config[:activation_function].apply_activation_function(@before_activation)
            end

            # call-seq:
            # backprop(Layer)
            #
            # Computes the @error of the layer using backpropagation formula.
            #
            # Backprop pass simplified is f'(before_activation) * next_layer.weighted_error.
            #
            def backprop next_layer
                @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(next_layer.weighted_error)
            end

            # Calls the optimizer set in the config.
            # Computes new optimized biases and weights, then sets them as the new weights and biases.
            #
            def optimize
                @biases = @config[:optimizer].optimize_biases(@biases, biases_gradient)
                @weights = @config[:optimizer].optimize_weights(@weights, weights_gradient)
            end

            # call-seq:
            # weighted_error -> Matrix
            #
            # Returns weighted error of layer.
            #
            def weighted_error
                @error * @weights
            end

            # call-seq:
            # weights_gradient -> Matrix
            #
            # Returns a Matrix containing the gradients of the weights.
            #
            def weights_gradient
                @error.t * @input / @output.row_size
            end

            # call-seq:
            # biases_gradient -> Vector
            #
            # Returns a Vector containing the gradients of the biases.
            #
            def biases_gradient
                #Sum error over batch, divide by @output.row_size to get average over batch
                Vector[*@error.t.map_row { |row| row.reduce(0) { |acc, val| acc + val } }] / @output.row_size
            end

            private

            # call-seq:
            # rand_weight -> Numeric
            #
            # Returns random Numeric between 0 and 1.
            #
            def rand_weight
                rand()
            end
        end

    end
end
