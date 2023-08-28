module Filo
    module Layer
        class InputLayer < Layer
            attr_reader :output

            def initialize size, config={}
                #Input layer has no input size
                super(nil, size, config)
            end

            def forward input
                #Check if each record in input matrix has the same size as the layer
                raise ShapeError.new unless input.column_size == @size

                @input = input
                @output = @input
            end
        end
    end
end
