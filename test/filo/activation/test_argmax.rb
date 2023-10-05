require_relative '../../test_helper'

class TestArgmax < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Argmax
    end

    def test_function
        input = Matrix[[2, -1, 0, 3], [-1, 0, 2, 0]]
        expected = Matrix[[0, 0, 0, 1], [0, 0, 1, 0]]

        output = @activation_function.apply_activation_function(input)

        assert_each_in_delta(expected, output)
    end

    def test_derivative
        assert_raise(StandardError) { @activation_function.apply_activation_function_derivative(Matrix[[1, 2, 3]]) }
    end
end
