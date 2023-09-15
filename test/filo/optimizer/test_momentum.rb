require_relative '../../test_helper'

class TestMomentum < Test::Unit::TestCase

    def setup
        @optimizer =  Filo::Optimizer.Momentum(learning_rate: 0.5, momentum: 0.8)
    end

    def test_optimize_biases
        biases = Vector[1]
        gradients = Vector[0.2]
        expected_after_first_timestep = Vector[0.9]
        expected_after_second_timestep = Vector[0.82]

        optimized_biases_first_timestep = @optimizer.optimize_biases(biases: biases, gradients: gradients)
        optimized_biases_second_timestep = @optimizer.optimize_biases(biases: biases, gradients: gradients)

        assert_each_in_delta(expected_after_first_timestep, optimized_biases_first_timestep, 0.001)
        assert_each_in_delta(expected_after_second_timestep, optimized_biases_second_timestep, 0.001)

    end

    def test_optimize_weights
        weights = Matrix[[1]]
        gradients = Matrix[[0.2]]
        expected_after_first_timestep = Matrix[[0.9]]
        expected_after_second_timestep = Matrix[[0.82]]

        optimized_weights_first_timestep = @optimizer.optimize_weights(weights: weights, gradients: gradients)
        optimized_weights_second_timestep = @optimizer.optimize_weights(weights: weights, gradients: gradients)

        assert_each_in_delta(expected_after_first_timestep, optimized_weights_first_timestep, 0.001)
        assert_each_in_delta(expected_after_second_timestep, optimized_weights_second_timestep, 0.001)
    end

    def test_optimize_weights_as_vector
        weights = Vector[1]
        gradients = Vector[0.2]
        expected_after_first_timestep = Vector[0.9]
        expected_after_second_timestep = Vector[0.82]

        optimized_weights_first_timestep = @optimizer.optimize_weights(weights: weights, gradients: gradients)
        optimized_weights_second_timestep = @optimizer.optimize_weights(weights: weights, gradients: gradients)

        assert_each_in_delta(expected_after_first_timestep, optimized_weights_first_timestep, 0.001)
        assert_each_in_delta(expected_after_second_timestep, optimized_weights_second_timestep, 0.001)
    end

end
