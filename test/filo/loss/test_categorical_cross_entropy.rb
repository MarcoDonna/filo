require_relative '../../test_helper'

class TestCategoricalCrossEntropy < Test::Unit::TestCase

    def setup
        @loss_function = Filo::Loss.CategoricalCrossEntropy
    end

    def test_loss_function
        predicted = Matrix[[0.2, 0.8], [0.6, 0.4]]
        observed = Matrix[[0, 1], [1, 0]]
        expected = [-Math.log(0.6), -Math.log(0.8)]

        loss = @loss_function.loss(predicted: predicted, observed: observed)

        assert_each_in_delta(expected, loss, 0.001)
    end

    def test_loss_derivative
        predicted = Matrix[[0.2, 0.8], [0.6, 0.4]]
        observed = Matrix[[0, 1], [1, 0]]
        expected = Matrix[[0, -1.25],[-1.6667, 0]]

        loss = @loss_function.loss_derivative(predicted: predicted, observed: observed)

        assert_each_in_delta(expected, loss, 0.001)
    end
end
