import katarchive
import time
import os

from subprocess import check_output

def put_test_inputfile(testdata_dir):
    prod = katarchive.search_archive(filename='1378901689.h5')[0]
    prod.download_dir = testdata_dir
    prod.path_to_file
    return prod.path_to_file

def build_analyse_self_generated_rfi_command(katfilename):
    force_system_python = '/usr/bin/python'
    my_exec = '/home/kat/RTS/svnScience/RTS/2.9-RFI/analyse_self_generated_rfi.py'
    return '%s %s %s' % (force_system_python, my_exec, katfilename)

#setup
testdata_dir = '/home/kat/RTS/test_area/Test_Data/'
test_inputfile = put_test_inputfile(testdata_dir)

if True:
    test_dir = '/home/kat/RTS/test_area/2.9-RFI/%i' % int(time.time())
else:
    test_dir =  '/home/kat/RTS/test_area/2.9-RFI/benchmark/'
os.mkdir(test_dir)

#execute
print 'Executing:'
cmd = build_analyse_self_generated_rfi_command(test_inputfile)
print '%s' % (cmd)
std_output = check_output(cmd, shell=True, cwd=test_dir)

#output
print 'Benchmark output viewable @:'
print 'http://sp-test.kat.ac.za/RTS/2.9-RFI/benchmark/'
print 'This run output viewable @:'
print '%s' % (test_dir.replace('/home/kat/RTS/test_area', 'http://sp-test.kat.ac.za/RTS'))

