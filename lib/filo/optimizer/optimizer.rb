module Filo
    module Optimizer

        class Optimizer

            def clear_internal_state what_to_clear=nil # :nodoc:
                return nil
            end

            def optimize_biases weights: nil, gradients: nil # :nodoc:
                raise MissingImplementationError.new
            end

            def optimize_weights weights: nil, gradients: nil # :nodoc:
                raise MissingImplementationError.new
            end
        end

    end
end
