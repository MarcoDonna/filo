module Filo
    module Activation
        #Parent class to all activation functions.
        #Each activation function inherits this class and implements the function and derivative methods
        class Activation

            def initialize config={}
                @config = config
            end

            #Apply activation function to each row of the matrix.
            #Applied to each row insted of single item is needed is some activation functions (softmax).
            def apply_activation_function matrix
                Matrix[*matrix.clone().to_a.map { |row| function(row) }]
            end

            # Apply activation function derivative to each row of the matrix.
            # Applied to each row insted of single item is needed is some activation functions (softmax).
            # The output is converted to Matrix only if the derivative is not jacobian
            def apply_activation_function_derivative matrix
                if(respond_to?(:jacobian?) and jacobian? === true)
                    matrix.clone().to_a.map { |row| derivative(row) }
                else
                    Matrix[*matrix.clone().to_a.map { |row| derivative(row) }]
                end
            end

            private

            def function vector
                raise MissingImplementationError.new
            end

            def derivative vector
                raise MissingImplementationError.new
            end
        end
    end
end
