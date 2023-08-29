require_relative '../../test_helper'

class TestOutputLayer < Test::Unit::TestCase
    def test_missing_loss_function
        assert_raise(StandardError) { Filo::Layer.OutputLayer(4, 2, {activation_function: Filo::Activation.Sigmoid})}
    end
end
