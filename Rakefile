require 'rake/testtask'
require 'rubygems/package_task'

require_relative 'lib/filo/version'

#https://ruby-doc.org/stdlib-3.1.0/libdoc/rake/rdoc/rake-13_0_6/doc/rakefile_rdoc.html
#https://docs.ruby-lang.org/en/2.1.0/Rake/TestTask.html
task :test => ['test:all']

namespace "test" do

    desc 'run all tests within the test folder (tests need to match the format test_*.rb)'
    Rake::TestTask.new :all do |t|
        t.libs << "test"
        t.test_files = FileList['test/filo/**/test_*.rb']
        t.verbose = true
    end

end

namespace :gem do

    desc 'package the gem'
    task :build => [:package]

    Gem::PackageTask.new(Gem::Specification.load('filo.gemspec')) do |pkg|
        pkg.need_tar = true # Specify if you want to generate a tar file
        pkg.need_zip = true # Specify if you want to generate a zip file
    end

    desc 'remove the pkg/ directory'
    task :clean do
        rm_rf 'pkg'
    end

    desc 'install the gem built from local source code'
    task :install => [:build] do
        cd 'pkg'
        # --local
        sh "gem install filo-#{Filo::VERSION}.gem"
        cd '..'
    end

end
