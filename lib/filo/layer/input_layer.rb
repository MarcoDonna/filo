module Filo
    module Layer

        #   input_layer = Filo::Layer.InputLayer(size: 2)
        #
        # Returns a new Intance of InputLayer.
        #
        def self.InputLayer size: nil
            return InputLayer.new(size: size)
        end

        class InputLayer < Layer
            attr_reader :output # :nodoc:

            def initialize size: nil # :notnew:
                raise ArgumentError.new if size.nil?

                # set @input_size to nil, input layer has no weights.
                @input_size = nil
                @size = size
            end

            # Sets @input and @output to both be equal to the matrix passed as input.
            #
            def forward input=nil
                raise ArgumentError.new if input.nil?
                raise ExceptionForMatrix::ErrDimensionMismatch.new unless input.column_size == @size

                @input = input
                @output = @input
                return @output
            end
        end

    end
end
