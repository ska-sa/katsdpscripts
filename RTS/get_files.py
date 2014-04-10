#!/usr/bin/python
import katarchive
import optparse
#iimport sys
#import subprocess
import os

def runProcess(exe):    
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(True):
      retcode = p.poll() #returns None while subprocess is running
      line = p.stdout.readline()
      yield line
      if(retcode is not None):
        break

parser = optparse.OptionParser(usage='%prog [options]',
                               description='This script finds datafiles to be reduced')
parser.add_option("-u", "--user", default='RTS',
                  help="the user who ran the obsevation, default %default)")
parser.add_option("-c", "--code",
                  help="Code number to look for")
parser.add_option( "-r","--red", default='echo',
                  help="Name of reduction script', default = '%default'")

(opts, args) = parser.parse_args()

files =  katarchive.search_archive(description='%s'%(opts.code),observer='RTS')
for f in files:
    f._get_archived_product()
    print f.metadata.Filename,f.metadata.Description
    if opts.red != 'echo' :
        os.system('%s %s'%(opts.red, f.path_to_file))
        #runProcess('%s %s'%(opts.red, f.path_to_file))
