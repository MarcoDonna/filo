module Filo
    module Optimizer

        class << self

            # call-seq:
            # Momentum(learning_rate: 0.01)
            #
            #   Filo::Optimizer.Momentum(learning_rate: 0.1)
            #
            # Returns a new instance of Momentum optimizer
            #
            def Momentum config={}
                Momentum.new(config)
            end
        end

        class Momentum < Optimizer # :nodoc:

            def initialize config # :notnew:
                super(config)
                @config[:learning_rate] = 0.01 if @config[:learning_rate].nil?
                @config[:momentum] = 0.9 if @config[:momentum].nil?

                @biases_momentum = nil
                @weights_momentum = nil
            end

            # call-seq:
            # optimize_vector(Vector, Vector) -> Vector
            #
            # Use Momentum to optimize a generic Vector of weights using a Vector of gradients
            def optimize_biases biases, gradients
                #delta = lr * gradient + momentum * momentum_factor
                #momentum = delta
                #bias = bias - delta
                @biases_momentum = Vector.zero(biases.size) if @biases_momentum.nil?

                delta = @config[:learning_rate] * gradients + @biases_momentum * @config[:momentum]
                @biases_momentum = delta
                return biases - delta
            end

            # call-seq:
            # optimize_vector(Vector, Vector) -> Vector
            #
            # Use Momentum to optimize a generic Vector of weights using a Vector of gradients or a Matrix of weights using a Matrix of gradients.
            #
            def optimize_weights weights, gradients
                case weights
                when Matrix
                    # TODO
                when Vector
                    @weights_momentum = Vector.zero(weights.size) if @weights_momentum.nil?

                    delta = @config[:learning_rate] * gradients + @weights_momentum * @config[:momentum]
                    @weights_momentum = delta
                    return weights - delta
                else
                    raise InvalidArgumentError.new
                end
            end
        end

    end
end
