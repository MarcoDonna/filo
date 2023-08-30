require_relative '../../test_helper'

class TestInvalidArgumentError < Test::Unit::TestCase
    def test_raise
        assert_raise(InvalidArgumentError) { raise InvalidArgumentError.new }
        assert_raise(InvalidArgumentError) { raise InvalidArgumentError.new() }
    end
end
