module Filo
    module Activation

        # call-seq:
        # Sigmoid(gain: 1) -> Filo::Activation::Sigmoid
        #
        # Returns a new instance of Sigmoid activation functions.
        #
        def self.Sigmoid gain: 1
            Sigmoid.new(gain: gain)
        end

        class Sigmoid < Activation # :nodoc:

            protected

            def initialize config
                @gain = config[:gain]
            end

            private

            def function vector
                return vector.map { |x| 1 / (1 + Math.exp(-x * @gain)) }
            end

            def derivative vector
                return function(vector).map { |x| @gain * x * (1 - x) }
            end
        end

    end
end
