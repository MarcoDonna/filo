require 'test/unit'
require 'matrix'

require_relative '../../lib/loss_function/mean_squared_error'

class TestMeanSquaredError < Test::Unit::TestCase
    def test_loss_function
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observerd = Matrix[[0, 1], [1, 0]]
        expected = [0.1, 0.04]

        loss_function = MeanSquaredError.new
        loss = loss_function.loss(predicted, observerd)

        assert_in_delta(0.001, expected[0], loss[0])
        assert_in_delta(0.001, expected[1], loss[1])
    end

    def test_loss_derivative
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observerd = Matrix[[0, 1], [1, 0]]
        expected = Matrix[[0.4, -0.4], [-0.8, -0.4]]
        expected = expected.to_a

        loss_function = MeanSquaredError.new
        loss_derivative = loss_function.loss_derivative(predicted, observerd).to_a

        assert_in_delta(0.001, expected[0][0].abs, loss_derivative[0][0].abs)
        assert_in_delta(0.001, expected[0][1].abs, loss_derivative[0][1].abs)
        assert_in_delta(0.001, expected[1][0].abs, loss_derivative[1][0].abs)
        assert_in_delta(0.001, expected[1][1].abs, loss_derivative[1][1].abs)
    end
end
