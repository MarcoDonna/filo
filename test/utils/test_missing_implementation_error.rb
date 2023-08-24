require 'test/unit'

require_relative '../../lib/utils/missing_implementation_error'

class TestMissingImplementation < Test::Unit::TestCase
    def test_raise
        assert_raise(MissingImplementationError) { raise MissingImplementationError.new }
        assert_raise(MissingImplementationError) { raise MissingImplementationError.new("Super useful message") }
    end
end
