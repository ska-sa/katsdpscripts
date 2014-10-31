from fabric.api import sudo, task, settings, env, hosts
import rts_common_deploy

#Set environment and hostnames
env.hosts = ['kat@10.98.4.2'] #['kat@192.168.6.174']
env.password = 'kat'

# Deb packages
DEB_PKGS = [ 'python-dev',                                                               #general
                    'gfortran libatlas-base-dev libblas-dev libexpat1-dev',
                    'git git-man',                                                       #git
                    'python-pip python-setuptools python-pkg-resources',                 #pip
                    'python-ply', 'python-twisted', 'python-unittest2', 'python-mock',   #for katcp
                    'subversion', 'nfs-kernel-server',
                    'ipython python-numpy python-scipy python-h5py',
                    'python-matplotlib python-pyfits python-pandas',                     #python stuff
                    'tree',
                    'ntp'
                    ]

# Pip packages
PIP_PKGS = ['scikits.fitting','pyephem','katcp','mplh5canvas','guppy','ProxyTypes','mod_pywebsocket']

# SKA git packages
SKA_GIT_PKGS = ['katpoint','katdal','PySPEAD','katsdpdisp']

@task
@hosts(env.hosts)
def deploy():
    """Example usage: fab rts_timeplot.deploy"""
    #configure for proxy access
    rts_common_deploy.site_proxy_configuration()

    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('apt-get -y update')

    #install deb packages: thin plooging
    # install ubuntu deb packages
    for pkg in DEB_PKGS: rts_common_deploy.install_deb_packages(pkg)

    #install pip packages: thin plooging
    # pip install python packages
    for pkg in PIP_PKGS: rts_common_deploy.install_pip_packages(pkg,flags='-U --no-deps')

    # install private ska-sa git packages
    for pkg in SKA_GIT_PKGS: rts_common_deploy.install_git_package(pkg,branch=rts_common_deploy.GIT_BRANCH)

    # setup ntp
    rts_common_deploy.ntp_configuration()


@task
@hosts(env.hosts)
def clear():
    # remove ska-sa git packages
    for pkg in reversed(SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)

    # pip uninstall python packages
    for pkg in reversed(PIP_PKGS + SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)
    # hack to remove scikits, which is still hanging around
    sudo('rm -rf /usr/local/lib/python2.7/dist-packages/scikits')

    # remove ubuntu deb packages
    for pkg in reversed(DEB_PKGS): rts_common_deploy.remove_deb_packages(pkg)

    sudo('apt-get -y autoremove')
