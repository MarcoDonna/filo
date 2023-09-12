class InvalidArgumentError < StandardError # :nodoc:
    def initialize message=nil
        super(message)
    end
end
