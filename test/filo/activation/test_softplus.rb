require_relative '../../test_helper'

class TestSoftplus < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Softplus
    end

    def test_function
        input = Matrix[[2, -1, 0]]
        expected = Matrix[[2.1269, 0.3133, 0.6931]]

        output = @activation_function.apply_activation_function(input)

        assert_each_in_delta(expected, output)
    end

    def test_derivative
        input = Matrix[[2, -1, 0]]
        expected = Matrix[[0.8808, 0.2689, 0.5]]

        output = @activation_function.apply_activation_function_derivative(input)

        assert_each_in_delta(expected, output)
    end
end
