module Filo
    module Loss

        class Loss # :nodoc:

            def loss predicted: nil, observed: nil
                raise MissingImplementationError.new
            end

            def loss_derivative predicted: nil, observed: nil
                raise MissingImplementationError.new
            end
        end

    end
end
