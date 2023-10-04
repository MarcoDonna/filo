require_relative '../../test_helper'

class TestRelu < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Relu
    end

    def test_function
        input = Matrix[[2, -3, 0]]
        expected = Matrix[[2, 0, 0]]

        output = @activation_function.apply_activation_function(input)

        assert_each_in_delta(expected, output)
    end

    def test_derivative
        input = Matrix[[2, -3, 0]]
        expected = Matrix[[1, 0, 0]]

        output = @activation_function.apply_activation_function_derivative(input)

        assert_each_in_delta(expected, output)
    end
end
