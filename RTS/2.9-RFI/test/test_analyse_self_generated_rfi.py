import katarchive
from subprocess import check_output

def put_test_inputfile(test_dir):
    prod = katarchive.search_archive(filename='1369128966.h5')[0]
    prod.download_dir = test_dir
    prod.path_to_file
    return prod.path_to_file

def build_analyse_self_generated_rfi_command(katfilename):
    force_system_python = '/usr/bin/python'
    my_exec = '/home/kat/RTS/svnScience/RTS/2.9-RFI/analyse_self_generated_rfi.py'
    return '%s %s %s' % (force_system_python, my_exec, katfilename)

#create output directory and get the test file
test_dir =  '/home/kat/RTS/test_area/2.9-RFI/1382008836/'
test_inputfile = put_test_inputfile(test_dir)

#execute in a shell in the test directory
cmd = build_analyse_self_generated_rfi_command(test_inputfile)
std_output = check_output(cmd, shell=True, cwd=test_dir)

#output viewable @ http://sp-test/RTS/2.9-RFI/blabla
