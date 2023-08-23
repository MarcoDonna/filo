require 'test/unit'
require 'byebug'

require_relative '../../src/utils/matrix'

class TestMatrix < Test::Unit::TestCase
    def test_each_row
        sum_of_rows = 0
        matrix = Matrix[[0, 1, 2, 3, 4, 5]]
        matrix = matrix.t
        matrix.each_row { |row| sum_of_rows += row[0] }

        assert_equal(15, sum_of_rows)
    end

    def test_map_row
        input = Matrix[[0, 1, 2, 3, 4, 5]]
        expected = [1, 2, 3, 4, 5, 6]

        input = input.t
        expected = expected

        output = input.map_row { |row| row[0] + 1 }

        assert_equal(expected, output)
    end
end
