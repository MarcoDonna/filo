module Filo
    module Activation

        def self.Relu
            return Relu.new
        end

        class Relu < Activation # :nodoc:

            def function(vector)
                return vector.map { |x| [0, x].max }
            end

            def derivative(vector)
                return vector.map { |x| x > 0 ? 1 : 0 }
            end

        end

    end
end
