class ShapeError < StandardError
    def initialize error_message=nil
        super(error_message)
    end
end