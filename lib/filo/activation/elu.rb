module Filo
    module Activation

        def self.Elu alpha: 1
            return Elu.new(alpha: alpha)
        end

        class Elu < Activation # :nodoc:

            def initialize alpha: 1
                @alpha = alpha
            end

            def function(vector)
                return vector.map { |x| x > 0 ? x : @alpha * (Math.exp(x) - 1) }
            end

            def derivative(vector)
                return vector.map { |x| x > 0 ? 1 : @alpha * Math.exp(x) }
            end

        end

    end
end
