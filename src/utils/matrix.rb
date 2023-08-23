require 'matrix'

class Matrix
    def each_row
        self.to_a.each { |row| yield row }
    end

    def map_row
        self.to_a.map { |row| yield row }
    end
end
