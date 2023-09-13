require 'test/unit'

require_relative '../lib/filo'

module Test::Unit::Assertions

    def assert_each_in_delta expected, actual, delta=0.001, message=""
        _wrap_assertion do
            pass = expected.to_a.flatten.zip(actual.to_a.flatten).reduce(true) { |acc, pair| acc and ((pair[0] - pair[1]).abs <= delta) }
            full_message = _assert_each_in_delta_message(expected, expected, actual, actual, delta, delta, message)

            assert_block(full_message) { pass }
        end
    end

    def _assert_each_in_delta_message expected, normalized_expected, actual, normalized_actual, delta, normalized_delta, message, options={}

        arguments = [expected, delta, actual]

        format = <<-EOT
        <?>
        +/-
        <?>
        was expected to include
        <?>
        EOT

        build_message(message, format, *arguments)
    end
end

