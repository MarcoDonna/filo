require 'rake/testtask'

#https://ruby-doc.org/stdlib-3.1.0/libdoc/rake/rdoc/rake-13_0_6/doc/rakefile_rdoc.html
#https://docs.ruby-lang.org/en/2.1.0/Rake/TestTask.html
task :test => ['test:all']

namespace "test" do

    Rake::TestTask.new :all  do |t|
        t.libs << "test"
        t.test_files = FileList['test/filo/**/test_*.rb']
        t.verbose = true
    end
    
end
