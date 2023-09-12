module Filo
    module Optimizer

        class << self

            # call-seq:
            # StochasticGradientDescent(learning_rate: 0.01)
            #
            #   Filo::Optimizer.StochasticGradientDescent(learning_rate: 0.1)
            #
            # Returns a new Instance of Stochastic Gradient Descent optimizer
            #
            def StochasticGradientDescent config={}
                StochasticGradientDescent.new(config)
            end
            alias_method :SGD, :StochasticGradientDescent

        end

        class StochasticGradientDescent < Optimizer # :nodoc:

            def initialize config # :notnew:
                super(config)
                @config[:learning_rate] = 0.01 if @config[:learning_rate].nil?
            end

            # call-seq:
            # optimize_vector(Vector, Vector) -> Vector
            #
            # Use SGD to optimize a generic Vector of weights using a Vector of gradients
            def optimize_biases biases, gradients
                #delta = -gradient * lr
                #bias = bias + delta
                optimize_vector(biases, gradients)
            end

            # call-seq:
            # optimize_vector(Vector, Vector) -> Vector
            #
            # Use SGD to optimize a generic Vector of weights using a Vector of gradients or a Matrix of weights using a Matrix of gradients.
            #
            def optimize_weights weights, gradients
                case  weights
                when Vector
                    optimize_vector(weights, gradients)
                when Matrix
                    weights + gradients * -@config[:learning_rate]
                else
                    raise InvalidArgumentError.new
                end
            end

            private

            # call-seq:
            # optimize_vector(Vector, Vector) -> Vector
            #
            # Use SGD to optimize a generic Vector of weights using a Vector of gradients
            def optimize_vector vec_w, vec_g
                vec_w + vec_g * -@config[:learning_rate]
            end
        end

    end
end
