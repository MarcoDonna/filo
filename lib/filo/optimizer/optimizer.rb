module Filo
    module Optimizer

        class Optimizer

            def initialize config={} # :notnew:
                @config = config
            end

            # Update the configuration of the optimizer after initialization
            #
            #   optimizer = Filo::Optimizer.SGD(learning_rate: 0.1)
            #   optimizer.update_config(learning_rate: 0.01)
            #
            def update_config new_config={}
                @config = @config.merge(new_config)
            end

            def optimize_vector vec_w, vec_g # :nodoc:
                raise MissingImplementationError.new
            end

            def optimize_biases biases, gradients # :nodoc:
                raise MissingImplementationError.new
            end

            def optimize_weights weights, gradients # :nodoc:
                raise MissingImplementationError.new
            end
        end

    end
end
