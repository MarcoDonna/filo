require 'test/unit'
require 'matrix'

require_relative '../../lib/utils/expanded_test_assertions'
require_relative '../../lib/loss_function/mean_squared_error'

class TestMeanSquaredError < Test::Unit::TestCase
    def test_loss_function
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observerd = Matrix[[0, 1], [1, 0]]
        expected = [0.1, 0.04]

        loss_function = MeanSquaredError.new
        loss = loss_function.loss(predicted, observerd)

        assert_each_in_delta(expected, loss, 0.001)
    end

    def test_loss_derivative
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observerd = Matrix[[0, 1], [1, 0]]
        expected = Matrix[[0.4, -0.4], [-0.8, -0.4]]

        loss_function = MeanSquaredError.new
        loss_derivative = loss_function.loss_derivative(predicted, observerd)

        assert_each_in_delta(expected.to_a.flatten, loss_derivative.to_a.flatten, 0.001)
    end
end
