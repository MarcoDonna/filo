require 'rake'

require_relative 'lib/filo'

Gem::Specification.new do |s|
    # Metadata
    s.name = 'filo'
    s.version = Filo::VERSION
    s.summary = 'Neural Networks with Ruby'
    s.description = 'Neural Networks library with Ruby'
    s.homepage = 'https://github.com/MarcoDonna/filo'
    s.licenses = ['Apache-2.0']

    # Authors and Maintainers
    s.authors = ['MarcoDonna']
    s.email = ['marco.donna03@gmail.com']

    # Requirements and dependencies
    s.required_ruby_version = '>= 3.0.0'
    s.add_dependency 'ractor', '~> 0.2.0'

    # Files and Directories
    # The files added in the gem are the same that have been addedto git (git ls-files -z), exclude some directories (use (example|test) to exlcude mutiple directories).
    s.files = FileList[Dir.chdir(File.expand_path(__dir__)) { `git ls-files -z`.split("\x0") }].exclude(/^(example)/)

    # Gem Specification Configuration
    # Specify the directory where executables files included in the gem should be installed
    s.bindir = 'bin'
    s.executables = FileList['bin/**/*'].to_a
    # Specifies the directories where Ruby files from your gem's library can be required from using 'require' within other ruby code
    s.require_paths = ['lib']
end
