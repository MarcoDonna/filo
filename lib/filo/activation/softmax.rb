module Filo
    module Activation

        # call-seq:
        # Softmax(config={}) -> Filo::Activation::Softmax
        #
        # Returns a new instance of Softmax activation functions.
        #
        def self.Softmax
            return Softmax.new
        end

        class Softmax < Activation # :nodoc:

            def jacobian?
                true
            end

            private

            def function vector
                # Calculate the unnormalized exponentials
                exp_values = vector.map { |x| Math.exp(x) }

                # Calculate the sum of exponentials
                exp_sum = exp_values.sum

                # Calculate softmax probabilities
                return exp_values.map { |exp| exp / exp_sum }.to_vector
            end

            def derivative vector
                # Calculate the softmax probabilities
                # https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
                softmax_probs = function(vector)

                # Calculate the Kronecker delta matrix
                delta_matrix = Array.new(softmax_probs.size) do |i|
                    Array.new(softmax_probs.size) do |j|
                    i == j ? softmax_probs[i] * (1 - softmax_probs[i]) : -softmax_probs[i] * softmax_probs[j]
                    end
                end

                return delta_matrix.to_matrix
            end
        end

    end
end
