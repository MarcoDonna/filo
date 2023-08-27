require 'matrix'

class Matrix


    def self.kronecker_delta diagonal_size
        Matrix[*Matrix.zero(diagonal_size).map_row { |row, idx| row[idx] = 1; row }]
    end

    def each_row
        self.to_a.each_with_index { |row, idx| yield row, idx }
    end

    def map_row
        self.to_a.map.each_with_index { |row, idx| yield row, idx }
    end
end
