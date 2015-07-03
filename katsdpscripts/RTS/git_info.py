import os
import sys
import warnings
try:
	import git
except:
	warnings.warn('gitpython package not found. Cannot provide repo information',ImportWarning)


def git_info(repo_location='/var/kat/katsdpscripts'):
	"""
	Helper function to get information about a github repo
	in a specified directory. For RTS this defaults to
	/var/kat/katsdpscripts but can be overridden by the user.
	"""

	git_repo = os.path.split(os.path.abspath(repo_location))[-1]
	repo_status=''
	try:
		repo = git.Repo(repo_location)
		git_time_stamp = repo.git.show(format='%ci',s=True)
		git_branch = repo.git.status().split('\n')[0].split(' ')[-1]
		git_hash = repo.git.show('--oneline').split()[0]
		if repo.git.status().split()[-1] != 'clean':
			repo_status = "-M"
	except:
		print 'Invalid Git repository at %s'%(repo_location)
		git_time_stamp,git_branch,git_hash = 'Unknown','Unknown','Unknown'


	return 'REPO: %s/%s%s#%s TIME: %s'%(git_repo,git_branch,repo_status,git_hash,git_time_stamp)

def iter_dir_tree(path):
	#Check for gitpython import
	try:
		git
	except NameError:
		return ''
	path = path.split('/')
	while not len(path) == 0:
		try :
			repo = git.Repo('/'.join(path) )
			break;
		except git.InvalidGitRepositoryError :
			path = path[0:-1]
			repo = None
	return '' if repo is None else '/'.join(path)

def get_git_path():
	"""
	Try and determine the base path of the github repo from which a script is run.
	Return empty string if it can't be found.
	"""
	path = iter_dir_tree(sys.argv[0])
	if path is '':
		path = iter_dir_tree(os.getcwd())
	return path