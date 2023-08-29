module Filo
    module Layer

        def self.OutputLayer input_size, size, config
            OutputLayer.new(config.merge(input_size: input_size, size: size))
        end

        class OutputLayer < DenseLayer
            attr_reader :loss_metric

            def initialize config={}
                super(config)
                raise StandardError.new("No :loss_function in layer config") if @config[:loss_function].nil?

                @loss_metric = []
            end

            def backprop target
                @target = target
                @loss_metric << @config[:loss_function].loss(@output, @target)
                @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(@output - @target)
            end
        end
        
    end
end
