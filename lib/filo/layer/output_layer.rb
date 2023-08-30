module Filo
    module Layer

        def self.OutputLayer *args
            if args.size == 1 and args[0].is_a?(Hash)
                OutputLayer.new(args[0])
            elsif args.size == 3 and args[2].is_a?(Hash)
                OutputLayer(args[2].merge(input_size: args[0], size: args[1]))
            else
                raise InvalidArgumentError.new
            end
        end

        class OutputLayer < DenseLayer
            attr_reader :loss_metric

            def initialize config={}
                super(config)
                raise InvalidArgumentError.new("No :loss_function in layer config") if @config[:loss_function].nil?

                @loss_metric = []
            end

            def backprop target
                raise ShapeError.new unless target.column_size == @output.column_size and target.row_size == @output.row_size

                @target = target
                @loss_metric << @config[:loss_function].loss(@output, @target)
                @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(@output - @target)
            end
        end

    end
end
