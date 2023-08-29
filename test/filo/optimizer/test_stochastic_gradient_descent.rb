require_relative '../../test_helper'

class TestStochasticGradientDescent < Test::Unit::TestCase

    def test_optimize_biases
        biases = Vector[1, -1, 0.5]
        gradients = Vector[0.8, 0.3, -0.2]
        expected = Vector[0.6, -1.15, 0.6]

        opt = Filo::Optimizer.StochasticGradientDescent({learning_rate: 0.5})
        optimized_biases = opt.optimize_biases(biases, gradients)

        assert_each_in_delta(expected.to_a, optimized_biases.to_a, 0.001)
    end

    def test_optimize_weights
        weights = Matrix[[1, -1, 0.5], [1, -1, 0.5]]
        gradients = Matrix[[0.8, 0.3, -0.2], [0.8, 0.3, -0.2]]
        expected = Matrix[[0.6, -1.15, 0.6], [0.6, -1.15, 0.6]]

        opt = Filo::Optimizer.StochasticGradientDescent({learning_rate: 0.5})
        optimized_weights = opt.optimize_weights(weights, gradients)

        assert_each_in_delta(expected.to_a.flatten,  optimized_weights.to_a.flatten, 0.001)
    end
end
