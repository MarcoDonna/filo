require_relative 'layer'
require_relative '../utils/matrix'
require_relative '../utils/shape_error'

class DenseLayer < Layer
    attr_reader :biases, :weights
    attr_reader :input, :before_activation, :output, :error

    def initialize input_size, size, config={}
        super(input_size, size, config)
        raise StandardError.new("No :activation_function in layer config") if @config[:activation_function].nil?

        @biases = Vector.zero(@size).map { rand_weight() }
        @weights = Matrix.build(@size, @input_size) { rand_weight() }
    end

    def biases= biases
        raise ShapeError.new unless biases.size == @size
        @biases = biases
    end

    def weights= weights
        raise ShapeError.new unless weights.column_size == @input_size and weights.row_size == @size
        @weights = weights
    end

    def forward prev_layer
        @input = prev_layer.output
        @before_activation = (@input * @weights.t).add_vector(@biases)
        @output = @config[:activation_function].apply_activation_function(@before_activation)
    end

    def backprop next_layer
        @error = @config[:activation_function].apply_activation_function_derivative(@before_activation).hadamard_product(next_layer.weighted_error)
    end

    def weighted_error
        @error * @weights
    end

    def weights_gradient
       @error.t * @input
    end

    def biases_gradient
        #Sum error over batch
        Vector[*@error.t.map_row { |row| row.reduce(0) { |acc, val| acc + val } }]
    end

    private

    def rand_weight
        rand()
    end
end
