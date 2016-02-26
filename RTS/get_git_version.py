#!/usr/bin/python
#Script to write out the current versions of installed github repos
#Used by the workflow manager which imports katsdpscripts at runtime and doesn't update
#__version__ when katsdpscripts is updated on the disk. This can be run by a subprocess
#call which will import katsdpscripts from a new environment and correctly report the 
#installed version.

from katsdpscripts import git_info
print git_info('standard')
