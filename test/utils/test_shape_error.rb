require 'test/unit'

require_relative '../../lib/utils/shape_error'

class TestShapeError < Test::Unit::TestCase
    def test_raise
        assert_raise(ShapeError) { raise ShapeError.new }
        assert_raise(ShapeError) { raise ShapeError.new("Super useful message") }
    end
end
