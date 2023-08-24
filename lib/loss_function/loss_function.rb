require_relative '../utils/missing_implementation_error'

#Parent class to all loss functions.
#Each loss function inherits this class and implements the loss and loss_derivative methods.
class LossFunction
    def initialize(config={})
        @config = config
    end

    #Should calculate the average loss across batch for each output neuron.
    def loss predicted_matrix, actual_matrix
        raise MissingImplementationError.new
    end

    def loss_derivative predicted_matrix, actual_matrix
        raise MissingImplementationError.new
    end
end
