require 'test/unit'

require_relative '../../lib/utils/matrix'
require_relative '../../lib/utils/shape_error'

class TestMatrix < Test::Unit::TestCase
    def test_kronecker_delta
        expected = Matrix[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        diag_size = expected.row_size

        matrix = Matrix.kronecker_delta(diag_size)
        assert_equal(expected, matrix)
    end

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

    def test_add_vec_row
        m = Matrix[[0, 1], [2, 3]]

        assert_raise(ShapeError) { m.add_vector(Vector[2])}
        assert_equal(Matrix[[1, 0], [3, 2]], m.add_vector(Vector[1, -1]))
    end

    def test_add_vec_column
        m = Matrix[[0, 1], [2, 3]]

        assert_raise(ShapeError) { m.add_vector(Vector[2], :column) }
        assert_equal(Matrix[[1, 1], [2, 2]], m.add_vector(Vector[1, -1], :column))
    end
end
