module Filo
    module Activation

        class Activation

            # call-seq:
            # apply_activation_function(Matrix) -> Matrix
            #
            # Applies +function+ to each row of matrix.
            #
            # This method is inherited and used by the activation functions, and each activation function implements it's
            # own +function+ method.
            #
            def apply_activation_function matrix
                return matrix.map_row { |row| function(row) }.to_matrix
            end

            # call-seq:
            # apply_activation_function_derivative(Matrix) -> Matrix
            # apply_activation_function_derivative(Matrix) -> Array
            #
            # Applies +derivative+ to each row of matrix.
            #
            # If the activation function is jacobian, returns an Array of the jacobian Matrices (one for each row).
            #
            # This method is inherited and used by the activation functions, and each activation function implements it's
            # own +derivative+ method.
            #
            def apply_activation_function_derivative matrix
                if(respond_to?(:jacobian?) and jacobian? === true)
                    return matrix.map_row { |row| derivative(row) }
                else
                    return matrix.map_row { |row| derivative(row) }
                end
            end

            # call-seq:
            # jacobian? -> boolean
            #
            # Return true if the derivative of the activation faction is jacobian, false if not.
            # This is set by each implementations of activation functions.
            #
            def jacobian?
                return false
            end

            private

            def function vector # :nodoc:
                raise MissingImplementationError.new
            end

            def derivative vector # :nodoc:
                raise MissingImplementationError.new
            end
        end

    end
end
