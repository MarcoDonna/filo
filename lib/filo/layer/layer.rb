module Filo
    module Layer
        class Layer
            attr_reader :size, :input_size

            def initialize config={}
                @config = config
                @input_size = @config[:input_size]
                @size = @config[:size]
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
