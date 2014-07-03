from fabric.api import sudo, task, hosts, settings, env
import rts_common_deploy

#Set environment and hostnames
env.hosts = ['kat@192.168.6.185']
env.password = 'kat'

# Deb packages for rts-imager
DEB_PKGS = [ 'python-dev',                                                        #general
                    'gfortran libatlas-base-dev libblas-dev libexpat1-dev',
                    'git git-man',                                                       #git
                    'python-pip python-setuptools python-pkg-resources',                 #pip
                    'libhdf5-dev',                                                       #h5py
                    'libpng12-dev libfreetype6-dev zlib1g-dev',                          #Matplotlib
                    'subversion', 'nfs-kernel-server',
                    'tree'
                    ]

# Pip packages for rts-imager
PIP_PKGS = ['ipython',
                   'numpy','scipy',                                                       #numpy/scipy
                   'h5py', 'scikits.fitting',
                   'matplotlib', 'pyfits']

# git packages for rts imager
SKA_GIT_PKGS = ['katpoint','katdal','katholog','katsdpscripts','katsdpworkflow','scape']

@task
def deploy():
    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get update')

    #install deb packages: thin plooging
    # install ubuntu deb packages
    for pkg in DEB_PKGS: rts_common_deploy.install_deb_packages(pkg)

    #install pip packages: thin plooging
    # pip install python packages
    for pkg in PIP_PKGS: rts_common_deploy.install_pip_packages(pkg)

        # install private ska-sa git packages
    for pkg in SKA_GIT_PKGS: rts_common_deploy.install_git_package(pkg,branch=rts_common_deploy.GIT_BRANCH)

    # oodt setup and install
    rts_common_deploy.oodt_setup()

@task
def clear():
    # remove oodt directories and packages
    rts_common_deploy.remove_oodt_directories()
    rts_common_deploy.remove_pip_packages('katoodt')

    # remove ska-sa git packages
    for pkg in reversed(SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)

    # pip uninstall python packages
    for pkg in reversed(PIP_PKGS + SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)
    # hack to remove scikits, which is still hanging around
    sudo('rm -rf /usr/local/lib/python2.7/dist-packages/scikits')

    # remove ubuntu deb packages
    for pkg in reversed(DEB_PKGS): rts_common_deploy.remove_deb_packages(pkg)
