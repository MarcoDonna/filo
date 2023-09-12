module Filo
    module Loss

        class Loss # :nodoc:

            def initialize(config={})
                @config = config
            end

            def loss predicted_matrix, actual_matrix
                raise MissingImplementationError.new
            end

            def loss_derivative predicted_matrix, actual_matrix
                raise MissingImplementationError.new
            end
        end

    end
end
