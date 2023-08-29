module Filo
    module Activation

        def self.Sigmoid config={}
            Sigmoid.new(config)
        end

        class Sigmoid < Activation
            #Sigmoid activation function
            #https://keisan.casio.com/exec/system/15157249643425

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
