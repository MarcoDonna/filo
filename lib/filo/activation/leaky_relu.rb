module Filo
    module Activation

        def self.LeakyRelu alpha: 0.01
            return LeakyRelu.new
        end

        class LeakyRelu < Activation # :nodoc:

            def initialize(alpha:0.01)
                @alpha = alpha
            end

            def function(vector)
                return vector.map { |x| x > 0 ? x : @alpha * x }
            end

            def derivative(vector)
                return vector.map { |x| x > 0 ? 1 : @alpha }
            end

        end

    end
end
