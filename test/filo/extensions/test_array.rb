require_relative '../../test_helper'

class TestArray < Test::Unit::TestCase

    def test_to_vector
        array = [1, 2, 3, 4, 5, 6]
        expected = Vector[1, 2, 3, 4, 5, 6]

        assert_equal(expected, array.to_vector)
    end

    def test_to_matrix
        array = [[1, 2], [3, 4], [5, 6]]
        expected = Matrix[[1, 2], [3, 4], [5, 6]]

        assert_equal(expected, array.to_matrix)
    end

end
