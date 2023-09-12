module Filo
    module Layer

        # call-seq:
        # InputLayer(size: Numeric) -> Filo::Layer::InputLayer
        # InputLayer(Numeric size, config={}) -> Filo::Layer::InputLayer
        #
        #   input_layer = Filo::Layer.InputLayer(2)
        #
        # Returns a new Intance of InputLayer.
        #
        def self.InputLayer *args
            if args[0].is_a?(Hash)
                InputLayer.new(args[0])
            elsif args[0].is_a?(Numeric)
                InputLayer({size: args[0]})
            else
                raise InvalidArgumentError.new
            end
        end

        class InputLayer < Layer
            attr_reader :output # :nodoc:

            # Set @input_size to nil, input layer has no weights.
            #
            def initialize config={} # :notnew:
                super(config)
                @input_size = nil
            end

            # call-seq:
            # forward(Matrix) -> Matrix
            #
            # Sets @input and @output to both be equal to the matrix passed as input.
            #
            def forward input
                #Check if each record in input matrix has the same size as the layer
                raise ShapeError.new unless input.column_size == @size

                @input = input
                @output = @input
            end
        end

    end
end
