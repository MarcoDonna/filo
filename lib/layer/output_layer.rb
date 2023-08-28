require_relative 'dense_layer'
require_relative '../utils/matrix'

class OutputLayer < DenseLayer
    attr_reader :loss_metric

    def initialize input_size, size, config={}
        super(input_size, size, config)
        raise StandardError.new("No :loss_function in layer config") if @config[:loss_function].nil?

        @loss_metric = []
    end

    def backprop target
        @target = target
        @loss_metric << @config[:loss_function].loss(@output, @target)
        @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(@output - @target)
    end
end
