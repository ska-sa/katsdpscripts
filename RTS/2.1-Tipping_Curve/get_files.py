#!/usr/bin/python
import katarchive
import optparse
parser = optparse.OptionParser(usage='%prog [options]',
                               description='This script finds datafiles to be reduced')
parser.add_option("-u", "--user", default='RTS',
                  help="the user who ran the obsevation, default %default)")
parser.add_option("-c", "--code",
                  help="Code number to look for")
parser.add_option( "-r","--red", default='echo ',
                  help="Name of reduction script', default = '%default'")

(opts, args) = parser.parse_args()

files =  katarchive.search_archive(description='%s'%(opts.code),observer='RTS')
print files
for f in files:
    f._get_archived_product()
    execfile('%s %s'%(opts.red,f.path_to_file))
