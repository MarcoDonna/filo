require_relative '../../test_helper'

class TestElu < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Elu
    end

    def test_function
        input = Matrix[[2, -1, 0]]
        expected = Matrix[[2, -0.632, 0]]

        output = @activation_function.apply_activation_function(input)

        assert_each_in_delta(expected, output)
    end

    def test_derivative
        input = Matrix[[2, -1, 0]]
        expected = Matrix[[1, 0.3679, 1]]

        output = @activation_function.apply_activation_function_derivative(input)

        assert_each_in_delta(expected, output)
    end
end
