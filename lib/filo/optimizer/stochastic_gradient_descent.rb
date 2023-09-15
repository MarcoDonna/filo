module Filo
    module Optimizer

        class << self

            #   Filo::Optimizer.StochasticGradientDescent(learning_rate: 0.1)
            #
            # Returns a new Instance of Stochastic Gradient Descent optimizer
            #
            def StochasticGradientDescent learning_rate: 0.01
                StochasticGradientDescent.new(learning_rate: learning_rate)
            end
            alias_method :SGD, :StochasticGradientDescent

        end

        class StochasticGradientDescent < Optimizer # :nodoc:

            def initialize learning_rate: 0.01 # :notnew:
                @learning_rate = learning_rate
            end

            # Use SGD to optimize a Vector of weights using a Vector of gradients
            #
            def optimize_biases biases: nil, gradients: nil
                raise ArgumentError.new if biases.nil? or gradients.nil?
                # delta = -gradient * lr
                # bias = bias + delta
                return biases + -@learning_rate * gradients
            end

            # Use SGD to optimize a Vector of weights using a Vector of gradients or a Matrix of weights using a Matrix of gradients.
            #
            def optimize_weights weights: nil, gradients: nil
                raise ArgumentError.new if weights.nil? or gradients.nil?

                case weights
                when Vector
                    return weights + -@learning_rate * gradients
                when Matrix
                    flattened_optimized_weights = optimize_weights(weights: weights.flatten, gradients: gradients.flatten)
                    return flattened_optimized_weights.unflatten(weights.width)
                else
                    raise ArgumentError.new
                end
            end

        end

    end
end
