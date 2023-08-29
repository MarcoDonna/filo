require_relative '../../test_helper'

class TestDenseLayer < Test::Unit::TestCase
    def test_missing_activation_function
        assert_raise(StandardError) { Filo::Layer.DenseLayer(3, 3) }
    end

    def test_set_parameters
        dense_layer = Filo::Layer.DenseLayer(3, 2, {activation_function: Filo::Activation.Sigmoid})

        new_biases = Vector[-1, 1]
        new_weights = Matrix[[-1, 0, 1], [-1, 0, 1]]

        dense_layer.biases = new_biases
        dense_layer.weights = new_weights

        assert_equal(new_biases, dense_layer.biases)
        assert_equal(new_weights, dense_layer.weights)

        assert_raise(ShapeError) { dense_layer.biases = Vector[-1, 0, 1] }
        assert_raise(ShapeError) { dense_layer.weights = Matrix[[-1, 0]]}
    end

    def test_forward
        input_layer = Filo::Layer.InputLayer(3)
        dense_layer = Filo::Layer.DenseLayer(3, 2, {activation_function: Filo::Activation.Sigmoid})

        input_matrix = Matrix[[0, 1, 2], [2, 3, 4], [4, 5, 6]]

        new_biases = Vector[-1, 1]
        new_weights = Matrix[[1, 1, 1], [1, 1, 1]]

        dense_layer.biases = new_biases
        dense_layer.weights = new_weights

        #Output of neuron 1 is the sum of each item in input vector -1
        #Output if neuron 2 is the sum of each item in input vector +1
        expected_before_activation = Matrix[[2, 4], [8, 10], [14, 16]]
        expected_output = Filo::Activation.Sigmoid.apply_activation_function(expected_before_activation)

        input_layer.forward(input_matrix)
        dense_layer.forward(input_layer)

        assert_equal(expected_before_activation, dense_layer.before_activation)
        assert_each_in_delta(expected_output.to_a.flatten, dense_layer.output.to_a.flatten)
    end
end
