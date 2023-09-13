require_relative '../../test_helper'

class TestSigmoid < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Sigmoid
        @activation_function_gain = Filo::Activation.Sigmoid(gain: 0.5)
    end

    def test_function
        input = Matrix[[-1, 0, 1], [-0.5, 0.2, 0.5]]
        expected = Matrix[[0.2689414213699951207488, 0.5, 0.7310585786300048792512], [0.3775406687981454, 0.5498339973124779085592, 0.6224593312018545646389]]
        expected_with_gain = Matrix[[0.3775406687981454, 0.5, 0.6224593312018546], [0.43782349911420193, 0.52497918747894, 0.5621765008857981]]

        output = @activation_function.apply_activation_function(input)
        output_with_gain = @activation_function_gain.apply_activation_function(input)

        assert_each_in_delta(expected, output)
        assert_each_in_delta(expected_with_gain, output_with_gain)
    end

    def test_derivative
        input = Matrix[[-1, 0, 1]]
        expected = Matrix[[0.1966119332414818525374, 0.25, 0.1966119332414818525374]]
        expected_with_gain = Matrix[[0.1175018561007972445347, 0.125, 0.1175018561007972445347]]

        output = @activation_function.apply_activation_function_derivative(input)
        output_with_gain = @activation_function_gain.apply_activation_function_derivative(input)

        assert_each_in_delta(expected_with_gain, output_with_gain)
        assert_each_in_delta(expected, output)
    end
end
