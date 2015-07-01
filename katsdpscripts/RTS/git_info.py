import os

def git_info(repo_location='/var/kat/katsdpscripts'):
	"""
	Helper function to get information about a github repo
	in a specified directory. For RTS this defaults to
	/var/kat/katsdpscripts but can be overridden by the user.
	"""

	git_repo = os.path.split(os.path.abspath(repo_location))[-1]
	repo_status=''
	try:
		import git
		repo = git.Repo(repo_location)
		git_time_stamp = repo.git.show(format='%ci',s=True)
		git_branch = repo.git.status().split('\n')[0].split(' ')[-1]
		git_hash = repo.git.show('--oneline').split()[0]
		if repo.git.status().split()[-1] != 'clean':
			repo_status = "-M"
	except:
		print 'Unable to get info on Git repository at %s'%(repo_location)
		git_time_stamp,git_branch,git_hash = 'Unknown','Unknown','Unknown'


	return 'REPO: %s/%s%s#%s TIME: %s'%(git_repo,git_branch,repo_status,git_hash,git_time_stamp)