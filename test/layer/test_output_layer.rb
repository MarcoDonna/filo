require 'test/unit'

require_relative '../../lib/utils/matrix'

require_relative '../../lib/layer/output_layer'
require_relative '../../lib/activation_function/sigmoid'

class TestOutputLayer < Test::Unit::TestCase

    def test_missing_loss_function
        assert_raise(StandardError) { OutputLayer.new(4, 2, {activation_function: Sigmoid.new})}
    end
end
