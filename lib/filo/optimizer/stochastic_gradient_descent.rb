module Filo
    module Optimizer
        class StochasticGradientDescent < Optimizer

            def initialize config
                super(config)
                @config[:learning_rate] = 0.01 if @config[:learning_rate].nil?
            end

            #Use SGD to optimize a Vector af biases using a Vector of gradients.
            def optimize_biases biases, gradients
                #delta = -gradient * lr
                #bias = bias + delta
                biases + gradients * -@config[:learning_rate]
            end

            #Use SGD top optimize a Matrix of weights using a Matrix of gradients
            def optimize_weights weights, gradients
                weights + gradients * -@config[:learning_rate]
            end
        end
    end
end
