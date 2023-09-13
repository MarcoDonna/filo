#--
# Copyright 2023 Marco Donna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#++

# Extend Matrix class with addictional functionalities, and maybe in the future GPGPU and Ractor support.
#
class Matrix

    RACTOR = false # :nodoc:
    RACTOR_CORES = 4 # :nodoc:

    alias_method :width, :column_size
    alias_method :height, :row_size

    # call-seq:
    # kronecker_delta(Numeric diagonal_size) -> Matrix
    #
    # Return a matrix with ones on the diagonal and zeros everywhere else.
    #
    def self.kronecker_delta diagonal_size
        Matrix[*Matrix.zero(diagonal_size).map_row { |row, idx| row[idx] = 1; row }]
    end

    # Transforms self to array and call each_with_index on each row.
    #
    def each_row
        self.to_a.each_with_index { |row, idx| yield row, idx }
    end

    # Transforms self to array and call map.each_with_index on each row.
    #
    def map_row
        self.to_a.map.each_with_index { |row, idx| yield row, idx }
    end

    # call-seq:
    # add_vector(Vector vec, direction=:row) -> Matrix
    # add_vector(Vector vec, :column) -> Matrix
    #
    # Sum each Vector in the self matrix with the Vector passed as argument.
    #
    # Direction can be specified to obtain row wise sum or column wise sum.
    #
    #   matrix = Matrix[[0, 1], [1, 2]]
    #   vector = Vector[10, 11]
    #   matrix.add_vector(vector) # => Matrix[[10, 12], [11, 13]]
    #
    def add_vector vec, direction=:row
        if direction === :row
            raise ShapeError.new unless self.column_size == vec.size
            return Matrix[*self.map_row { |row| Vector[*row] + vec }]
        elsif direction === :column
            self.t.add_vector(vec)
        else
            raise InvalidArgumentError.new
        end
    end

    def flatten
        return self.to_a.flatten.to_vector
    end

end

class Vector # :nodoc:

    alias_method :old_plus, :+

    def +(v)
        case v
        when Vector
            if Matrix::RACTOR === true
                # Use Ractor or GPGPU here to extend the class Vector
                # Only Proof of concept, Ractor still experimental (01/09/2023)
                # and the addition is already quick, this code runs slower then standatd implementation.
                # Could be useful somewhere else, where the core operations is longer then a simple sum of 2 numbers
                # In the following example code, the Ractor implementation is much faster then no parallel code

                #   def rand_init(n)
                #       n.times.map { |i|  sleep(0.001); rand }
                #   end

                #   def ractor_rand_init(n, cores=1)
                #       pll = cores.times.map do |i|
                #           Ractor.new(i, n/cores.to_f) do |i, n|
                #               Ractor.yield rand_init(n.to_i)
                #           end
                #       end
                #
                #       pll.map { |p| p.take }.flatten
                #   end
                #
                #   n = 5000
                #   Benchmark.bm do |x|
                #   x.report { rand_init(n) }
                #   x.report { ractor_rand_init(n, 4) }
                #   end

                # Split self and v in Matrix::RACTOR_CORES batches
                size = (v.size / Matrix::RACTOR_CORES.to_f).ceil
                self_split = self.each_slice(size).to_a.map { |x| x.nil? ? [] : Vector[*x] }.flatten
                v_split = v.each_slice(size).to_a.map { |x| x.nil? ? [] : Vector[*x] }.flatten

                workers = Matrix::RACTOR_CORES.times.map do |i|
                    Ractor.new(self_split[i], v_split[i]) do |a, b|
                        Ractor.yield(a.old_plus(b)) unless a.nil?
                    end
                end

                results = Matrix::RACTOR_CORES.times.map do |i|
                    workers[i].take
                end

                results = Vector[*results.map(&:to_a).flatten]
                return results
            else
                old_plus(v)
            end
        else
            old_plus(v)
        end
    end

    def outer_product vector
        return self.to_a.product(vector.to_a).collect { |a, b| a * b }.to_vector.unflatten(size())
    end

    def unflatten columns
        return self.each_slice(columns).to_a.to_matrix
    end

end
