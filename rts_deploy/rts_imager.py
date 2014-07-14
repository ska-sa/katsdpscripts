from fabric.api import sudo, task, hosts, settings, env
from fabric.contrib import files

from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package, remove_deb_packages
from rts_common_deploy import make_directory #, remove_dir
from rts_common_deploy import deploy_oodt_comp_ver_06, deploy_solr, configure_tomcat
from rts_common_deploy import STAGING_HOME, STAGING_INGEST, STAGING_FAILED, ARCHIVE_DATA, SOLR_COLLECTIONS_HOME, RTS_DATA, OODT_HOME, OODT_CONF, VAR_KAT
from rts_common_deploy import GIT_BRANCH
import os

#Set environment and hostnames
env.hosts = ['kat@192.168.6.185']
env.password = 'kat'

#rts-imager specific data areas
STAGING_AREA = '/data/staging_area'
PROCESS_AREA = '/data/process_area'

#oodt flavoured directories specific to imager machine
WORKFLOW_AREA = '/var/kat/katsdpworkflow'
STAGING_NFS_INGEST = os.path.join(STAGING_HOME, "nfs_rts")

#log directories that need kat ownership
CELERY_LOG = '/var/log/celery'
TOMCAT7_LOG = '/var/log/tomcat7'

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
             'tomcat7', 'nfs-common']

# Pip packages for rts-imager
PIP_PKGS = ['pyephem', 'scikits.fitting']

# SKA private git packages for rts imager
SKA_PRIVATE_GIT_PKGS = ['katpoint', 'katdal', 'katholog', 'katsdpscripts', 'scape']

def mount_datadir():
    make_directory('/data')
    files.append('/etc/fstab',
                 'UUID=88f7342e-177d-4a9d-af18-b7b669335412 /data ext4  defaults 0 0',
                 use_sudo=True)
    sudo('mount -a')

def configure_celery():
    sudo('/etc/init.d/celeryd stop')
    remove_deb_packages('python-librabbitmq') #known celery bug
    CELERYD_CONF='/etc/default/celeryd'
    files.sed(CELERYD_CONF,
                'ENABLED="false"',
                'ENABLED="true"',
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
                '#CELERYD_OPTS="--time-limit=300 --concurrency=8"',
                'CELERYD_OPTS="--concurrency=1',
                use_sudo=True)
    sudo('/etc/init.d/celeryd start')

def nfs_mount_archive_dir():
    pass
    
def make_directory_trees():
    make_directory(VAR_KAT)
    make_directory(OODT_HOME)
    make_directory(OODT_CONF)
    make_directory(WORKFLOW_AREA)
    make_directory(STAGING_INGEST)
    make_directory(STAGING_NFS_INGEST)
    make_directory(STAGING_FAILED)
    make_directory(ARCHIVE_DATA)
    make_directory(SOLR_COLLECTIONS_HOME)
    make_directory(RTS_DATA)
    make_directory(STAGING_AREA)
    make_directory(PROCESS_AREA)
    make_directory(CELERY_LOG) #change owner
    make_directory(TOMCAT7_LOG) #change owner
    

def deploy_oodt():
    deploy_oodt_comp_ver_06("cas-filemgr")
    deploy_oodt_comp_ver_06("cas-crawler")
    deploy_solr()
    configure_tomcat()

@task
@hosts(env.hosts)
def deploy():
    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('apt-get -y update')
    
    #install deb packages: thin plooging
    # install ubuntu deb packages
    for pkg in DEB_PKGS: install_deb_packages(pkg)
    
    #install pip packages: thin plooging
    # pip install python packages
    for pkg in PIP_PKGS: install_pip_packages(pkg,flags='-U --no-deps')
    
    # install private ska-sa git packages
    for pkg in SKA_PRIVATE_GIT_PKGS: install_git_package(pkg, branch=GIT_BRANCH)
    
    mount_datadir()
    make_directory_trees()
    deploy_oodt()
    # pip katsdpworkflow and oodt configuration in its final resting place
    retrieve_git_package('oodt_conf',output_location=OODT_CONF)
    retrieve_git_package('katsdpworkflow',output_location=WORKFLOW_AREA)
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
