module Filo
    module Loss

        class << self

            #   loss_function = Filo::Loss.MeanSquaredError
            #
            # Returns a new instance of Mean Squared Error loss functions
            #
            def MeanSquaredError
                return MeanSquaredError.new
            end
            alias_method :MSE, :MeanSquaredError

        end

        class MeanSquaredError < Loss # :nodoc:

            # Calculate the average loss across batch for each output neuron. Using Mean Squared Error.
            #
            def loss predicted: nil, observed: nil
                raise ArgumentError.new if predicted.nil? or observed.nil?

                return (predicted - observed).t.map_row do |row|
                    row.inject(0) { |acc, val| acc + val**2 } / row.size
                end
            end

            # For each batch item and each output neuron, there will be a corresponding derivative value.
            #
            def loss_derivative predicted: nil, observed: nil
                raise ArgumentError.new if predicted.nil? or observed.nil?

                result_as_array = (predicted - observed).map_row { |row| row.map { |x| 2 * x }}
                return result_as_array.to_matrix
            end
        end

    end
end
