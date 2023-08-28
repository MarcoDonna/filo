module Filo
    module Optimizer
        class Optimizer

            def initialize config={}
                @config = config
            end

            def optimize_biases biases, gradients
                raise MissingImplementationError.new
            end

            def optimize_weights weights, gradients
                raise MissingImplementationError.new
            end
        end
    end
end
