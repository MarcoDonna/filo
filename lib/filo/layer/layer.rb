module Filo
    module Layer
        class Layer

            def initialize input_size, size, config={}
                @input_size = input_size
                @size = size
                @config = config
            end

            #Input can be Matrix or Layer
            def forward input
                raise MissingImplementationError.new
            end

            #Can backprop deltas or layer
            def backprop error
                raise MissingImplementationError.new
            end
        end
    end
end
