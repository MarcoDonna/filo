require_relative '../../test_helper'

class TestInputLayer < Test::Unit::TestCase
    def test_shape_error
        input = Matrix[[2, 3, 4], [5, 6, 7]]

        layer = Filo::Layer.InputLayer(4)
        assert_raise(ShapeError) { layer.forward(input) }

        layer = Filo::Layer.InputLayer(2)
        assert_raise(ShapeError) { layer.forward(input) }
    end

    def test_forward
        input = Matrix[[2, 3, 4], [5, 6, 7]]
        expected = Matrix[[2, 3, 4], [5, 6, 7]]

        layer = Filo::Layer.InputLayer(3)
        layer.forward(input)

        assert_equal(expected, layer.output)
        assert_true(input.equal?(layer.output))
    end

    def test_backprop_fail
        layer = Filo::Layer.InputLayer(2)
        assert_raise(MissingImplementationError) { layer.backprop(nil) }
    end
end
