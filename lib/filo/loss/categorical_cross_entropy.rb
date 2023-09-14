module Filo
    module Loss

        class << self

            #   loss_function = Filo::Loss.CategoricalCrossEntropy
            #
            # Returns a new instance of Categorical Cross Entropy loss functions
            #
            def CategoricalCrossEntropy
                CategoricalCrossEntropy.new
            end
            alias_method :CCE, :CategoricalCrossEntropy

        end

        class CategoricalCrossEntropy < Loss # :nodoc:

            # Calculate the average loss across batch for each output neuron. Using Categorical Cross Entropy.
            #
            def loss predicted: nil, observed: nil
                raise ArgumentError.new if predicted.nil? or observed.nil?

                log_predicted = predicted.map_row { |row| row.map { |x| Math.log(x) }}
                element_wise = log_predicted.to_matrix.hadamard_product(observed)

                return element_wise.t.map_row { |row| -row.sum }
            end

            # For each batch item and each output neuron, there will be a corresponding derivative value.
            #
            def loss_derivative predicted: nil, observed: nil
                raise ArgumentError.new if predicted.nil? or observed.nil?

                inverse_predicted = predicted.map_row { |row| row.map { |x| -1.0/x }}
                return inverse_predicted.to_matrix.hadamard_product(observed)
            end
        end

    end
end
