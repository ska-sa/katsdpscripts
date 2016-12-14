import os
from subprocess import call
from tempfile import mkdtemp
import argparse

import nbformat

def build_ipynb_obs_report_command(datafile,template):
    obs_report_template_filename = template
    filename_base = os.path.basename(datafile)
    obs_report_output_filename = 'obs_report_%s.ipynb'%filename_base
    file_in = open(obs_report_template_filename)
    file_out = open(obs_report_output_filename,'w')
    nb = nbformat.reader.read(file_in)
    # the format of ipython notebooks changed ... find out the version and treat accordingly
    ipy_version = int(nb["nbformat"])-1
    if ipy_version == 2:
        for sheet in nb["worksheets"]:
            for cell in sheet["cells"]:
                if cell['input'].startswith('filename ='):
                    cell['input'] = "filename = '%s'" % datafile
    else:
        for cell in nb["cells"]:
            if cell['source'].startswith('filename ='):
                cell['source'] = "filename = '%s'" % datafile
    nbformat.write(nb,file_out)
    file_in.close()
    file_out.close()
    return 'runipy -o -s %s' % obs_report_output_filename

def generate_ipynb_obs_report(datafile,template):
    # call obs report
    obs_report_command = build_ipynb_obs_report_command(datafile,template)
    print 'Command executed:'
    print obs_report_command
    # todo: we'll want to know know why the returncode was not 0
    call(obs_report_command, shell=True)
    print 'jupyter nbconvert --to html --template basic %s'%obs_report_command.split()[-1]
    call('jupyter nbconvert --to html --template basic %s'%obs_report_command.split()[-1], shell=True)
    
parser = argparse.ArgumentParser(description="Runs the docker ipython obs-report on a file")
parser.add_argument("filename", nargs=1)
parser.add_argument("--template", default='/var/kat/katsdpscripts/AR1/obs_report/AR1_obs_report_template.ipynb')
args,unknown = parser.parse_known_args()

if args.filename[0] == '':
    raise RuntimeError('Please specify an h5 file to load.')
if not os.path.exists(args.template):
    raise IOError('Ipython template not found at %s.'%(args.template,)) 
generate_ipynb_obs_report(args.filename[0],args.template)
