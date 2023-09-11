module Filo
    module Loss

        class << self

            def CategoricalCrossEntropy config={}
                CategoricalCrossEntropy.new(config)
            end

            alias_method :CCE, :CategoricalCrossEntropy

        end

        class CategoricalCrossEntropy < Loss

            #Calculate the average loss across batch for each output neuron.
            def loss predicted_matrix, observed_matrix
                log_predicted_matrix = Matrix[*predicted_matrix.map_row { |row| row.map { |x| Math.log(x) }}]
                log_predicted_matrix.hadamard_product(observed_matrix).t.map_row do |row|
                    -row.inject(0) { |acc, val| acc + val } / row.size
                end
            end

            #For each batch item and each output neuron, there will be a corresponding derivative value.
            def loss_derivative predicted_matrix, observed_matrix
                inverse_predicted = Matrix[*predicted_matrix.map_row { |row| row.map { |x| -1.0/x }}]
                observed_matrix.hadamard_product(inverse_predicted)
            end
        end

    end
end
