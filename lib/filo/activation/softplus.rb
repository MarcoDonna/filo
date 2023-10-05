module Filo
    module Activation

        def self.Softplus
            return Softplus.new
        end

        class Softplus < Activation # :nodoc:

            def function(vector)
                return vector.map { |x| Math.log(1 + Math.exp(x)) }
            end

            def derivative(vector)
                return vector.map { |x| 1 / (1 + Math.exp(-x)) }
            end

        end

    end
end
