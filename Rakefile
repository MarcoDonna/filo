require 'rake/testtask'

#https://ruby-doc.org/stdlib-3.1.0/libdoc/rake/rdoc/rake-13_0_6/doc/rakefile_rdoc.html
#https://docs.ruby-lang.org/en/2.1.0/Rake/TestTask.html
task :test => 'test:all'

namespace "test" do
    #task :all => [:utils, :activation_functions, :loss_functions]

    Rake::TestTask.new :all  do |t|
        t.libs << "test"
        t.test_files = FileList['test/**/test_*.rb']
        t.verbose = true
    end

    Rake::TestTask.new :utils  do |t|
        t.libs << "test"
        t.test_files = FileList['test/utils/**/test_*.rb']
        t.verbose = true
    end

    Rake::TestTask.new :activation_functions  do |t|
        t.libs << "test"
        t.test_files = FileList['test/activation_function/**/test_*.rb']
        t.verbose = true
    end

    Rake::TestTask.new :loss_functions  do |t|
        t.libs << "test"
        t.test_files = FileList['test/loss_function/**/test_*.rb']
        t.verbose = true
    end

    Rake::TestTask.new :optimizers  do |t|
        t.libs << "test"
        t.test_files = FileList['test/optimizer/**/test_*.rb']
        t.verbose = true
    end

    Rake::TestTask.new :layers  do |t|
        t.libs << "test"
        t.test_files = FileList['test/layer/**/test_*.rb']
        t.verbose = true
    end
end


