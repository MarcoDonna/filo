require_relative '../../test_helper'

class TestMeanSquaredError < Test::Unit::TestCase

    def setup
        @loss_function = Filo::Loss.MeanSquaredError
    end

    def test_loss_function
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observed = Matrix[[0, 1], [1, 0]]
        expected = [0.1, 0.04]

        loss = @loss_function.loss(predicted: predicted, observed: observed)

        assert_each_in_delta(expected, loss, 0.001)
    end

    def test_loss_derivative
        predicted = Matrix[[0.2, 0.8], [0.6, -0.2]]
        observed = Matrix[[0, 1], [1, 0]]
        expected = Matrix[[0.4, -0.4], [-0.8, -0.4]]

        loss_derivative = @loss_function.loss_derivative(predicted: predicted, observed: observed)

        assert_each_in_delta(expected, loss_derivative, 0.001)
    end
end
