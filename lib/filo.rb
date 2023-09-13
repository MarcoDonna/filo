require 'matrix'
require 'ractor'

require_relative 'filo/version'

require_relative 'filo/extensions/matrix'
require_relative 'filo/extensions/array'

require_relative 'filo/utils/missing_implementation_error'
require_relative 'filo/utils/invalid_argument_error'
require_relative 'filo/utils/shape_error'

require_relative 'filo/activation'
require_relative 'filo/layer'
require_relative 'filo/loss'
require_relative 'filo/optimizer'
require_relative 'filo/neural_network'

module Filo

end
