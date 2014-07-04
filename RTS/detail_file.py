#!/usr/bin/python
import katdal
import optparse
parser = optparse.OptionParser(usage='%prog [options]  <filename>',
                               description='This details  datafiles to be reduced')

(opts, args) = parser.parse_args()

h5 = katdal.open(args[0])
print h5

for  x in h5.file['History/script_log']: print x[1]
