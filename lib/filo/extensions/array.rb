class Array

    def to_vector
        return Vector[*self]
    end

    def to_matrix
        return Matrix[*self]
    end

end
