module Filo
    module Activation

        # call-seq:
        # Sigmoid(gain: 1) -> Filo::Activation::Sigmoid
        #
        # Returns a new instance of Sigmoid activation functions.
        #
        def self.Sigmoid config={}
            Sigmoid.new(config)
        end

        class Sigmoid < Activation # :nodoc:

            def initialize config={}
                super(config)
                @config[:gain] = 1 if @config[:gain].nil?
            end

            private

            def function vector
                vector.map { |x| 1 / (1 + Math.exp(-x * @config[:gain])) }
            end

            def derivative vector
                function(vector).map { |x| @config[:gain] * x * (1 - x) }
            end
        end

    end
end
