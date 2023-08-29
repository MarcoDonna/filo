module Filo
    module Loss

        class << self
            
            def MeanSquaredError config={}
                MeanSquaredError.new(config)
            end

            alias_method :MSE, :MeanSquaredError

        end

        class MeanSquaredError < Loss

            #Calculate the average loss across batch for each output neuron.
            def loss predicted_matrix, observed_matrix
                (predicted_matrix - observed_matrix).t.map_row do |row|
                    row.inject(0) { |acc, val| acc + val**2 } / row.size
                end
            end

            #For each batch item and each output neuron, there will be a corresponding derivative value.
            def loss_derivative predicted_matrix, observed_matrix
                Matrix[*(predicted_matrix - observed_matrix).map_row { |row| row.map { |x| 2 * x } }]
            end
        end

    end
end
