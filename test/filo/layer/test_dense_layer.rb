require_relative '../../test_helper'

class TestDenseLayer < Test::Unit::TestCase

    def setup
        @dense_layer = Filo::Layer.DenseLayer(input_size: 3,
                                              size: 2,
                                              activation_function: Filo::Activation.Sigmoid,
                                              optimizer: Filo::Optimizer.SGD(learning_rate: 0.5))
        @input_layer = build_input_layer()
        @next_layer = build_next_layer()

        @biases = Vector[-1, 1]
        @weights = Matrix[[-1, 0, 1], [1, 0, 1]]

        @dense_layer.biases = @biases
        @dense_layer.weights = @weights

        @input_matrix = Matrix[[1, 2, 3]]
    end

    def build_input_layer
        return Filo::Layer.InputLayer(size: 3)
    end

    def build_next_layer
        return Filo::Layer.DenseLayer(input_size: 2, size: 1, activation_function: Filo::Activation.Sigmoid)
    end

    def test_set_parameters_fail
        assert_raise(ShapeError) { @dense_layer.biases = Vector[-1, 0, 1] }
        assert_raise(ShapeError) { @dense_layer.weights = Matrix[[-1, 0]]}
    end

    def test_forward
        expected_before_activation = Matrix[[1, 5]]
        expected_output = Filo::Activation.Sigmoid.apply_activation_function(expected_before_activation)

        @input_layer.forward(@input_matrix)
        @dense_layer.forward(@input_layer)

        assert_each_in_delta(expected_before_activation, @dense_layer.before_activation)
        assert_each_in_delta(expected_output, @dense_layer.output)
    end

    def test_backprop
        expected_error = Matrix[[-0.24751657271185995, 0.2139096965202944]]
        @next_layer.error = Matrix[[1]]
        @next_layer.weights = Matrix[[-1, 1]]
        @dense_layer.before_activation = Matrix[[0.2, 0.8]]

        assert_each_in_delta(expected_error, @dense_layer.backprop(@next_layer))
    end

    def test_weighted_error
        error = Matrix[[0, 1], [-1, 0], [3, 1]]
        expected = Matrix[[1, 0, 1], [1, 0, -1], [-2, 0, 4]]

        @dense_layer.error = error

        assert_each_in_delta(expected, @dense_layer.weighted_error)
    end

    def test_weights_gradient
        error = Matrix[[0, 1]]
        input = Matrix[[1, 2]]
        # only used to get the bach size
        output = Matrix[[1, 2, 3]]
        expected = Matrix[[0, 0], [1, 2]]

        @dense_layer.error = error
        @dense_layer.output = output
        @dense_layer.input = input

        assert_each_in_delta(expected, @dense_layer.weights_gradient)
    end

    def test_biases_gradient
        error = Matrix[[0, 1], [-1, 0], [3, 1]]
        # only used to get the bach size
        output = Matrix[[1, 2, 3]]
        # average error for neuron
        expected = Vector[2, 2]

        @dense_layer.error = error
        @dense_layer.output = output

        assert_each_in_delta(expected, @dense_layer.biases_gradient)
    end
end
