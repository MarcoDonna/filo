#Copyright 2023 Marco Donna
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

require 'matrix'
require 'ractor'

class Matrix
    RACTOR = false
    RACTOR_CORES = 4

    alias_method :width, :column_size
    alias_method :height, :row_size

    def self.kronecker_delta diagonal_size
        Matrix[*Matrix.zero(diagonal_size).map_row { |row, idx| row[idx] = 1; row }]
    end

    def each_row
        self.to_a.each_with_index { |row, idx| yield row, idx }
    end

    def map_row
        self.to_a.map.each_with_index { |row, idx| yield row, idx }
    end

    def add_vector vec, direction=:row
        if direction == :row
            raise ShapeError.new unless self.column_size == vec.size
            return Matrix[*self.map_row { |row| Vector[*row] + vec }]
        else
            self.t.add_vector(vec)
        end
    end
end

class Vector

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

end
