from fabric.api import sudo, task, hosts, settings, env
from fabric.contrib import files

from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package, remove_deb_packages
from rts_common_deploy import make_directory, check_and_make_sym_link, auto_start_filemgr
from rts_common_deploy import deploy_oodt_comp_ver_06
from rts_common_deploy import OODT_HOME, OODT_CONF, VAR_KAT #, RTS_DATA, ARCHIVE_DATA, STAGING_HOME, STAGING_INGEST, STAGING_FAILED, SOLR_COLLECTIONS_HOME, 
from rts_common_deploy import GIT_BRANCH

#Set environment and hostnames
env.hosts = ['kat@192.168.6.185']
env.password = 'kat'

#rts-imager specific data areas
STAGING_AREA = '/data/staging_area'
PROCESS_AREA = '/data/process_area'

#area to put katsdpscripts
SCRIPTS_AREA = '/var/kat/katsdpscripts'

#oodt flavoured directories specific to imager machine
WORKFLOW_AREA = '/var/kat/katsdpworkflow'

#log directories that need kat ownership
CELERY_LOG = '/var/log/celery'
TOMCAT7_LOG = '/var/log/tomcat7'
CAS_FILEMGR_LOG = '/var/log/cas_filemgr'

# Deb packages for rts-imager
DEB_PKGS = [ 'vim', 'python-dev',                                                 #general
             'gfortran', 'libatlas-base-dev', 'libblas-dev', 'libexpat1-dev',
             'git', 'git-man',                                                    #git
             'python-pip', 'python-setuptools', 'python-pkg-resources',           #pip
             'subversion', 'nfs-kernel-server',
             'python-celery', 'celeryd', 'rabbitmq-server',
             'tree', 'pyflakes', 'openjdk-7-jre', 'htop',                         #Other things
             'ipython', 'python-numpy', 'python-scipy', 'python-h5py',
             'python-matplotlib', 'python-pyfits', 'python-pandas', 'python-nose',#python stuff
             'nfs-common']#'tomcat7', 'nfs-common']

# Pip packages for rts-imager
PIP_PKGS = ['pyephem', 'scikits.fitting', 'pysolr']

# SKA private git packages for rts imager
SKA_PRIVATE_GIT_PKGS = ['katpoint', 'katdal', 'katholog', 'scape', 'katsdpdata']

def deploy_oodt():
    deploy_oodt_comp_ver_06("cas-filemgr")

def make_directory_trees():
    make_directory(VAR_KAT)
    make_directory(OODT_HOME)
    make_directory(OODT_CONF)
    make_directory(WORKFLOW_AREA)
    make_directory(STAGING_AREA)
    make_directory(PROCESS_AREA)
    make_directory(CELERY_LOG) #change owner
    make_directory(CAS_FILEMGR_LOG)
    make_directory('/data')
    make_directory('/export/archive/data')
    make_directory('/home/kat/.config/matplotlib')

def auto_mounts():
    files.append('/etc/fstab',
                 'UUID=88f7342e-177d-4a9d-af18-b7b669335412 /data ext4 defaults 0 0',
                 use_sudo=True)
    files.append('/etc/fstab',
                 '192.168.1.7:/export/archive/data /export/archive/data nfs _netdev,rw,soft,intr,auto,tcp,bg 0 0',
                 use_sudo=True)
    sudo('mount -a')
    check_and_make_sym_link('/export/archive/data', '/var/kat/archive/data'), 

def configure_celery():
    sudo('/etc/init.d/celeryd stop')
    remove_deb_packages('python-librabbitmq') #known celery bug
    CELERYD_CONF='/etc/default/celeryd'
    files.sed(CELERYD_CONF,
                'ENABLED="false"',
                'ENABLED="true"',
                use_sudo=True)
    files.sed(CELERYD_CONF,
                'CELERYD_NODES="w1"',
                'CELERYD_NODES="rts-imager"',
                use_sudo=True)
    files.append(CELERYD_CONF,
                'CELERY_APP="katsdpworkflow"',
                use_sudo=True)
    files.sed(CELERYD_CONF,
                'CELERYD_USER="celery"',
                'CELERYD_USER="kat"',
                use_sudo=True)
    files.sed(CELERYD_CONF,
                'CELERYD_GROUP="celery"',
                'CELERYD_GROUP="kat"',
                use_sudo=True)
    files.sed(CELERYD_CONF,
                'CELERYD_CHDIR="/opt/Myproject/"',
                'CELERYD_CHDIR="/var/kat/katsdpworkflow/"',
                use_sudo=True)
    files.sed(CELERYD_CONF,
                'CELERYD_OPTS="--time-limit=300 --concurrency=8"',
                'CELERYD_OPTS="--concurrency=1"',
                use_sudo=True)
    sudo('/etc/init.d/celeryd start')

def configure_matplotlib():
    files.append('/home/kat/.config/matplotlib/matplotlibrc',
                   'backend:Agg')
@task
@hosts(env.hosts)
def deploy():
    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('umount /export/archive/data')
        sudo('umount /data')
        sudo('apt-get -y update')
    
    # install ubuntu deb packages
    for pkg in DEB_PKGS: install_deb_packages(pkg)
    
    #install pip packages: thin plooging
    # pip install python packages
    for pkg in PIP_PKGS: install_pip_packages(pkg,flags='-U --no-deps')
    
    # install private ska-sa git packages
    for pkg in SKA_PRIVATE_GIT_PKGS: install_git_package(pkg, branch=GIT_BRANCH)
    
    make_directory_trees()
    auto_mounts()
    deploy_oodt()
    # pip katsdpworkflow and oodt configuration in its final resting place
    retrieve_git_package('oodt_conf', output_location=OODT_CONF)
    retrieve_git_package('katsdpworkflow', output_location=WORKFLOW_AREA)

    # retrieve katsdpscripts and install (need the RTS scripts in a locateable area)
    retrieve_git_package('katsdpscripts', output_location=SCRIPTS_AREA)
    install_pip_packages(SCRIPTS_AREA, flags='-U --no-deps')

    auto_start_filemgr()
    configure_matplotlib()
    configure_celery()

# @task
# @hosts(env.hosts)
# def clear():
#     # remove staging and processing dirs
#     rts_common_deploy.remove_dir(STAGING_AREA)
#     rts_common_deploy.remove_dir(PROCESS_AREA)
# 
#     # remove oodt directories and packages
#     rts_common_deploy.remove_oodt_directories()
#     rts_common_deploy.remove_pip_packages('katoodt')
# 
#     # remove ska-sa git packages
#     for pkg in reversed(SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)
# 
#     # pip uninstall python packages
#     for pkg in reversed(PIP_PKGS + SKA_GIT_PKGS): rts_common_deploy.remove_pip_packages(pkg)
#     # hack to remove scikits, which is still hanging around
#     sudo('rm -rf /usr/local/lib/python2.7/dist-packages/scikits')
# 
#     # remove ubuntu deb packages
#     for pkg in reversed(DEB_PKGS): rts_common_deploy.remove_deb_packages(pkg)
# 
#     sudo('apt-get -y autoremove')
