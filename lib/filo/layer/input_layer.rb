module Filo
    module Layer

        def self.InputLayer size
            InputLayer.new({size: size})
        end

        class InputLayer < Layer
            attr_reader :output

            def initialize config={}
                #Input layer has no input size
                super(config)
                @input_size = nil
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
