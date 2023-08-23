require 'test/unit'
require 'matrix'

require_relative '../../src/activation_function/sigmoid'

class TestSigmoid < Test::Unit::TestCase
    def test_function
        input = Matrix[[-1, 0, 1], [-0.5, 0.2, 0.5]]
        expected = Matrix[[0.2689414213699951207488, 0.5, 0.7310585786300048792512], [0.3775406687981454, 0.5498339973124779085592, 0.6224593312018545646389]]
        expected_with_gain = Matrix[[0.3775406687981454, 0.5, 0.6224593312018546], [0.43782349911420193, 0.52497918747894, 0.5621765008857981]]

        af = Sigmoid.new
        output = af.apply_activation_function(input)
        assert_equal(expected, output)

        af = Sigmoid.new({gain: 0.5})
        output = af.apply_activation_function(input)
        assert_equal(expected_with_gain, output)

        #This is effectively pointer comparison.
        #The method apply_activation_function creates a new instance of Matrix as output.
        #Input and output should not share pointer.
        assert_false(input.equal?(output))
    end

    def test_derivative
        input = Matrix[[-1, 0, 1]]
        expected = Matrix[[0.1966119332414818525374, 0.25, 0.1966119332414818525374]]
        expected_with_gain = Matrix[[0.1175018561007972445347, 0.125, 0.1175018561007972445347]]

        af = Sigmoid.new
        output = af.apply_activation_function_derivative(input)
        assert_equal(expected, output)

        af = Sigmoid.new({gain: 0.5})
        output = af.apply_activation_function_derivative(input)
        assert_equal(expected_with_gain, output)

        #This is effectively pointer comparison.
        #The method apply_activation_function creates a new instance of Matrix as output.
        #Input and output should not share pointer.
        assert_false(input.equal?(output))
    end
end
