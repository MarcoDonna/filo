module Filo
    module Layer

        # call-seq:
        # OutputLayer(input_size: Numeric, size: Numeric, activation_function: Activation, loss_function: Loss, optimizer: Optimzer=nil) -> Filo::Layer::OutputLayer
        # OutputLayer(Numeric input_size, Numeric size, config={}) -> Filo::Layer::OutputLayer
        #
        #   output_layer = Filo::Layer.OutputLayer(2, 4, activation_function: Filo::Activation.Sigmoid, loss_function: Filo::Loss.MSE)
        #   output_layer = Filo::Layer.OutputLayer(input_size: 2, size: 4, activation_function: Filo::Activation.Sigmoid, loss_function: Filo::Loss.MSE)
        #
        # Returns a new Intance of OutputLayer.
        #
        def self.OutputLayer size: nil, input_size: nil, activation_function: nil, loss_function: nil, optimizer: nil
            return OutputLayer.new(size: size,
                                   input_size: input_size,
                                   activation_function: activation_function,
                                   loss_function: loss_function,
                                   optimizer: optimizer)
        end

        class OutputLayer < DenseLayer
            attr_reader :loss_metric

            def initialize size: nil, input_size: nil, activation_function: nil, loss_function: nil, optimizer: nil # :notnew:
                super(size: size, input_size: input_size, activation_function: activation_function, optimizer: optimizer)

                raise ArgumentError.new if loss_function.nil?

                @loss_function = loss_function
                @loss_metric = []
            end

            # call-seq:
            # backprop(Layer)
            #
            # Computes the @error of the layer using backpropagation formula and the loss function derivative.
            #
            # Loss function derivative can be both a Matrix or jacobian (an Array of Matrix).
            #
            def backprop target
                raise ShapeError.new unless target.column_size == @output.column_size and target.row_size == @output.row_size

                @target = target
                @loss_metric << @loss_function.loss(predicted: @output, observed: @target)
                loss_gradient = @loss_function.loss_derivative(predicted: @output, observed: @target)
                # Check if the derivative of the activation function is a Vector (sigmoid) or a Matrix (softplus)
                # Somet things need to change, maybe have method of the activation functions that tells us what the expected output is
                if(@activation_function.respond_to?(:jacobian?) and @activation_function.jacobian? === true)
                    # Softmax like activation function derivative
                    applied_activation_derivative = @activation_function.apply_activation_function_derivative(@before_activation)
                    # Each item inside applied_activation_derivative is a Matrix, each Matrix is the derivative of one item in the batch.
                    # For each Matrix, perform matrix-vector multiplication to compute the error of the output layer
                    @error = applied_activation_derivative.map.with_index { |matrix, i| matrix * Vector[*loss_gradient.to_a[i]] }.to_matrix
                else
                    # Sigmoid like activation function derivative
                    @error = @activation_function.apply_activation_function_derivative(@before_activation).hadamard_product(loss_gradient)
                end
            end
        end

    end
end
