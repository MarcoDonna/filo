module Filo
    module Layer

        class Layer # :nodoc:
            attr_reader :size, :input_size

            def forward input
                raise MissingImplementationError.new
            end

            def backprop error
                raise MissingImplementationError.new
            end
        end

    end
end
