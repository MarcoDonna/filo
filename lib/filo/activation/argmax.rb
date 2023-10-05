module Filo
    module Activation

        def self.Argmax
            return Argmax.new
        end

        class Argmax < Activation # :nodoc:

            def function(vector)
                max_index = vector.each_with_index.max[1]
                ret = Vector.zero(vector.size)
                ret[max_index] = 1
                return ret
            end

            def derivative(vector)
                raise StandardError.new('Argmax has no useful derivative')
            end

        end

    end
end
