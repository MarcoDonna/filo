require_relative '../../test_helper'

class TestInputLayer < Test::Unit::TestCase

    def setup
        @valid_layer = Filo::Layer.InputLayer(size: 3)
        @invalid_layer = Filo::Layer.InputLayer(size: 4)

        @input_matrix = Matrix[[2, 3, 4], [5, 6, 7]]
    end

    def test_shape_error
        assert_raise(ExceptionForMatrix::ErrDimensionMismatch) { @invalid_layer.forward(@input_matrix)}
    end

    def test_forward
        expected = Matrix[[2, 3, 4], [5, 6, 7]]

        output = @valid_layer.forward(@input_matrix)

        assert_equal(output, @valid_layer.output)
        assert_equal(expected, output)
    end

end
