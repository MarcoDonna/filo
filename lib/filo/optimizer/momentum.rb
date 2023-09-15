module Filo
    module Optimizer

        class << self

            #   Filo::Optimizer.Momentum(learning_rate: 0.1, momentum: 0.8)
            #
            # Returns a new instance of Momentum optimizer
            #
            def Momentum learning_rate: 0.01, momentum: 0.9
                Momentum.new(learning_rate: learning_rate, momentum: momentum)
            end
        end

        class Momentum < Optimizer # :nodoc:

            def initialize learning_rate: 0.01, momentum: 0.9 # :notnew:
                @learning_rate = learning_rate
                @momentum = momentum

                clear_internal_state()
            end

            # Reset internal variables used by the optimizer
            #
            def clear_internal_state what_to_clear=nil
                case what_to_clear
                when :weights
                    @weights_momentum = nil
                when :biases
                    @biases_momentum = nil
                else
                    clear_internal_state(:weights)
                    clear_internal_state(:biases)
                end
            end

            # Use Momentum to optimize a generic Vector of weights using a Vector of gradients
            #
            def optimize_biases biases: nil, gradients: nil
                raise ArgumentError.new if biases.nil? or gradients.nil?

                # Create a buffer to store the changes at the previous timestep
                @biases_momentum = Vector.zero(biases.size) if @biases_momentum.nil?

                # delta_t = momentum * delta_(t-1) + gradient
                # save delta_t in the buffer, will be used at next timestep
                # bias = bias - learningrate * delta_t
                delta = @learning_rate * gradients + @biases_momentum * @momentum
                @biases_momentum = delta

                return biases - delta
            end

            # Use Momentum to optimize a generic Vector of weights using a Vector of gradients or a Matrix of weights using a Matrix of gradients.
            #
            def optimize_weights weights: nil, gradients: nil
                raise ArgumentError.new if weights.nil? or gradients.nil?

                case weights
                when Vector
                    @weights_momentum = Vector.zero(weights.size) if @weights_momentum.nil?

                    delta = @learning_rate * gradients + @weights_momentum * @momentum
                    @weights_momentum = delta

                    return weights - delta
                when Matrix
                    flattened_optimized_weights = optimize_weights(weights: weights.flatten, gradients: gradients.flatten)
                    return flattened_optimized_weights.unflatten(weights.width)
                else
                    raise InvalidArgumentError.new
                end
            end
        end

    end
end
