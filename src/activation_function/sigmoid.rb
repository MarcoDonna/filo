require_relative 'activation_function'

#Sigmoid activation function
#https://keisan.casio.com/exec/system/15157249643425
class Sigmoid < ActivationFunction
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

