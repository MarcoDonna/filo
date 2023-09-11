module Filo
    module Layer

        def self.OutputLayer *args
            if args.size == 1 and args[0].is_a?(Hash)
                OutputLayer.new(args[0])
            elsif args.size == 3 and args[2].is_a?(Hash)
                OutputLayer(args[2].merge(input_size: args[0], size: args[1]))
            else
                raise InvalidArgumentError.new
            end
        end

        class OutputLayer < DenseLayer
            attr_reader :loss_metric

            def initialize config={}
                super(config)
                raise InvalidArgumentError.new("No :loss_function in layer config") if @config[:loss_function].nil?

                @loss_metric = []
            end

            def backprop target
                raise ShapeError.new unless target.column_size == @output.column_size and target.row_size == @output.row_size

                @target = target
                @loss_metric << @config[:loss_function].loss(@output, @target)
                loss_gradient = @config[:loss_function].loss_derivative(@output, @target)
                # Check if the derivative of the activation function is a Vector (sigmoid) or a Matrix (softplus)
                # Somet things need to change, maybe have method of the activation functions that tells us what the expected output is
                if(@config[:activation_function].respond_to?(:jacobian?) and @config[:activation_function].jacobian? === true)
                    # Softmax like activation function derivative
                    applied_activation_derivative = @config[:activation_function].apply_activation_function_derivative(@before_activation)
                    # Each item inside applied_activation_derivative is a Matrix, each Matrix is the derivative of one item in the batch.
                    # For each Matrix, perform matrix-vector multiplication to compute the error of the output layer
                    @error = Matrix[*applied_activation_derivative.map.with_index { |matrix, i| matrix * Vector[*loss_gradient.to_a[i]] }]
                else
                    # Sigmoid like activation function derivative
                    @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(loss_gradient)
                end
            end

            def optimize
                @biases = @config[:optimizer].optimize_weights(@biases, biases_gradient)
                @weights = @config[:optimizer].optimize_weights(@weights, weights_gradient)
            end

        end

    end
end
