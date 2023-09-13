require_relative '../../test_helper'

class TestSoftmax < Test::Unit::TestCase

    def setup
        @activation_function = Filo::Activation.Softmax
    end

    def test_function
        input = Matrix[[1.0, 2.0, 3.0], [1.5, 0.5, 2.0]]
        expected = Matrix[[0.09003, 0.24473, 0.66524], [0.33122, 0.12187, 0.54691]]

        output = @activation_function.apply_activation_function(input)
        assert_each_in_delta(expected, output)
    end

    def test_derivative
        input = Matrix[[1.0, 2.0, 3.0], [1.5, 0.5, 2.0]]
        expected = [
            Matrix[[0.08259, -0.02204, -0.06055], [-0.02204, 0.18816, -0.16612], [-0.06055, -0.16612, 0.22167]],
            Matrix[[0.22175, -0.04055, -0.18120], [-0.04055, 0.09963, -0.05808], [-0.18120, -0.05808, 0.24992]]
        ]

        output = @activation_function.apply_activation_function_derivative(input)
        assert_each_in_delta(expected[0], output[0], 0.01)
        assert_each_in_delta(expected[1], output[1], 0.01)
    end
end
