#!/usr/bin/env python

import os
import argparse
import nbformat
from subprocess import call
from tempfile import mkdtemp

def build_ipynb_bf_folded_obs_report_command(directory, template):
    print directory
    bf_obs_report_template_filename = template
    directory_base = os.path.basename(directory)
    bf_obs_report_output_filename = 'bf_folded_obs_report_%s.ipynb' % directory_base
    file_in = open(bf_obs_report_template_filename)
    file_out = open(bf_obs_report_output_filename, 'w')
    nb = nbformat.reader.read(file_in)
    # the format of ipython notebooks changed... find out the version and treat accordingly
    ipy_version = int(nb['nbformat']) - 1
    if ipy_version == 2:
        for sheet in nb['worksheets']:
            for cell in sheet['cells']:
                if cell['input'].startswith('directory ='):
                    cell['input'] = "directory = '%s'" % directory
    else:
        for cell in nb['cells']:
            if cell['source'].startswith('directory ='):
                cell['source'] = "directory = '%s'" % directory
    nbformat.write(nb, file_out)
    file_in.close()
    file_out.close()
    return 'runipy --overwrite --skip-exceptions %s' % bf_obs_report_output_filename

def generate_ipynb_bf_folded_obs_report(directory, template):
    # call obs report
    obs_report_command = build_ipynb_bf_obs_report_command(directory, template)
    print 'Command executed:'
    print obs_report_command
    # todo: we'll want to know know why the returncode was not 0
    call(obs_report_command, shell=True)
    print 'jupyter nbconvert --to html --template full %s' % obs_report_command.split()[-1]
    call('jupyter nbconvert --to html --template full %s' % obs_report_command.split()[-1], shell=True)

parser = argparse.ArgumentParser(description='Runs the docker for IPython beamformer observation report.')
parser.add_argument('directory', nargs=1)
parser.add_argument('--template', dest='template', default='/home/kat/software/katsdpscripts/reduction/obs_report/bf_folded_obs_report_template.ipynb')
args, unknown = parser.parse_known_args()

if args.directory[0] == '':
    raise RuntimeError('Please specify a directory from which files will be loaded.')
if not os.path.isfile(args.template):
    raise IOError('IPython template not found at %s.' % args.template)
generate_ipynb_bf_folded_obs_report(args.directory[0], args.template)
