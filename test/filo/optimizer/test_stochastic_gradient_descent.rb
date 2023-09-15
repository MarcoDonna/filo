require_relative '../../test_helper'

class TestStochasticGradientDescent < Test::Unit::TestCase

    def setup
        @optimizer =  Filo::Optimizer.StochasticGradientDescent(learning_rate: 0.5)
    end

    def test_optimize_biases
        biases = Vector[1, -1, 0.5]
        gradients = Vector[0.8, 0.3, -0.2]
        expected = Vector[0.6, -1.15, 0.6]

        optimized_biases = @optimizer.optimize_biases(biases: biases, gradients: gradients)

        assert_each_in_delta(expected, optimized_biases, 0.001)
    end

    def test_optimize_weights
        weights = Matrix[[1, -1, 0.5], [1, -1, 0.5]]
        gradients = Matrix[[0.8, 0.3, -0.2], [0.8, 0.3, -0.2]]
        expected = Matrix[[0.6, -1.15, 0.6], [0.6, -1.15, 0.6]]

        optimized_weights = @optimizer.optimize_weights(weights: weights, gradients: gradients)

        assert_each_in_delta(expected,  optimized_weights, 0.001)
    end

    def test_optimize_weights_as_vector
        weights = Vector[1, -1, 0.5]
        gradients = Vector[0.8, 0.3, -0.2]
        expected = Vector[0.6, -1.15, 0.6]

        optimized_weights = @optimizer.optimize_weights(weights: weights, gradients: gradients)

        assert_each_in_delta(expected,  optimized_weights, 0.001)
    end

end
