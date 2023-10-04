require_relative '../../test_helper'

class TestLeakyRelu < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.LeakyRelu
        @activation_function_with_alpha = Filo::Activation.LeakyRelu(alpha: 0.01)
    end

    def test_function
        input = Matrix[[2, -3, 0]]
        expected = Matrix[[2, -0.03, 0]]

        output = @activation_function.apply_activation_function(input)

        assert_each_in_delta(expected, output)
    end

    def test_derivative
        input = Matrix[[2, -3, 0]]
        expected = Matrix[[1, 0.01, 0.01]]

        output = @activation_function.apply_activation_function_derivative(input)

        assert_each_in_delta(expected, output)
    end
end
