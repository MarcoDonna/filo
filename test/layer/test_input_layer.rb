require 'test/unit'
require 'matrix'

require_relative '../../lib/layer/input_layer'
require_relative '../../lib/utils/shape_error'
require_relative '../../lib/utils/missing_implementation_error'

class TestInputLayer < Test::Unit::TestCase
    def test_shape_error
        input = Matrix[[2, 3, 4], [5, 6, 7]]

        layer = InputLayer.new(4)
        assert_raise(ShapeError) { layer.forward(input) }

        layer = InputLayer.new(2)
        assert_raise(ShapeError) { layer.forward(input) }
    end

    def test_forward
        input = Matrix[[2, 3, 4], [5, 6, 7]]
        expected = Matrix[[2, 3, 4], [5, 6, 7]]

        layer = InputLayer.new(3)
        layer.forward(input)

        assert_equal(expected, layer.output)
        assert_true(input.equal?(layer.output))
    end

    def test_backprop_fail
        layer = InputLayer.new(2)
        assert_raise(MissingImplementationError) { layer.backprop(nil) }
    end
end
