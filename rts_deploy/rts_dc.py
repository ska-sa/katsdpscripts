from fabric.api import sudo, task, hosts, settings, env
from fabric.contrib import files

# from fabric.api import sudo, run, env, task, cd, settings
# from fabric.contrib import files
import os
from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package
from rts_common_deploy import deploy_oodt_comp_ver_06, deploy_solr, configure_tomcat, auto_start_filemgr, auto_start_crawler_rts
from rts_common_deploy import make_directory, check_and_make_sym_link , site_proxy_configuration
from rts_common_deploy import ntp_configuration
from rts_common_deploy import OODT_HOME, OODT_CONF
from rts_common_deploy import GIT_BRANCH

#Set environment and hostnames
env.hosts = ['kat@10.98.2.1']
env.password = 'kat'

# Deb packages for rts-imager
DEB_PKGS = [ 'vim', 'python-dev',                                                 #general
             'gfortran', 'libatlas-base-dev', 'libblas-dev', 'libexpat1-dev',
             'git', 'git-man',                                                    #git
             'python-pip', 'python-setuptools', 'python-pkg-resources',           #pip
             'python-ply', 'python-twisted', 'python-unittest2', 'python-mock', 
             'subversion', 'nfs-kernel-server',
             'python-celery', 'celeryd', 'rabbitmq-server',
             'tree', 'pyflakes', 'openjdk-7-jre', 'htop',                         #Other things
             'ipython', 'python-numpy', 'python-scipy', 'python-h5py',
             'python-matplotlib', 'python-pyfits', 'python-pandas', 'python-nose',#python stuff
             'nfs-common', 'tomcat7', 'nfs-common', 'ntp']

# Pip packages for rts-dc
PIP_PKGS = ['pyephem', 'scikits.fitting', 'pysolr', 'katcp', 'ProxyTypes']

# SKA private git packages for rts imager
SKA_PRIVATE_GIT_PKGS = ['katpoint', 'katdal', 'katholog', 'scape', 'PySPEAD', 'katsdpdata', 'katsdpcontroller', 'katsdpingest']

TOMCAT7_LOG = '/var/log/tomcat7'
CAS_FILEMGR_LOG = '/var/log/cas_filemgr'
CAS_CRAWLER_LOG = '/var/log/cas_crawler'

# oodt install info
VAR_KAT = '/var/kat'
OODT_HOME = '/usr/local/oodt/'
STAGING_HOME = '/var/kat/data'
ARCHIVE_HOME = '/var/kat/archive'
SDP_MC = '/var/kat/sdpcontroller'
# Subdirectories derived from roots
# Where we're going to put the oodt configuration files
OODT_CONF = os.path.join(OODT_HOME, "conf")
# Primary staging directory
SOLR_COLLECTIONS_HOME = os.path.join(ARCHIVE_HOME, "catalogs/solr")
# Staging dir
STAGING_INGEST = os.path.join(STAGING_HOME, "staging")
STAGING_FAILED = os.path.join(STAGING_HOME, "failed")
STAGING_NFS_INGEST = os.path.join(STAGING_HOME, "nfs_staging")
# Archive directory where all the data products will be stored
ARCHIVE_DATA = os.path.join(ARCHIVE_HOME, "data")

def make_directory_trees():
    make_directory(VAR_KAT)
    make_directory(OODT_HOME)
    make_directory(CAS_FILEMGR_LOG)
    make_directory(CAS_CRAWLER_LOG)
    make_directory('/export/RTS/')
    make_directory(STAGING_HOME)
    make_directory(ARCHIVE_HOME)
    make_directory(SDP_MC)
    make_directory(SOLR_COLLECTIONS_HOME)
    make_directory(STAGING_INGEST)
    make_directory(STAGING_FAILED)
    make_directory(STAGING_NFS_INGEST)
    make_directory(ARCHIVE_DATA)
    make_directory(OODT_CONF)

def deploy_oodt():
    deploy_oodt_comp_ver_06("cas-crawler")
    deploy_oodt_comp_ver_06("cas-filemgr")
    retrieve_git_package('oodt_conf', output_location=OODT_CONF)
    files.sed(OODT_CONF+'/cas-filemgr/etc/filemgr.properties',
                'org.apache.oodt.cas.filemgr.catalog.solr.url=http://192.168.1.50:8983/solr',
                'org.apache.oodt.cas.filemgr.catalog.solr.url=http://127.0.0.1:8983/solr')
    deploy_solr()
    configure_tomcat()

def auto_mounts():
    files.append('/etc/fstab',
                 'kat-archive.karoo.kat.ac.za:/mnt/md3000i/sci_proc/RTS /export/RTS/ nfs4  _netdev,rw,soft,intr,auto,tcp,bg 0 0',
                 use_sudo=True)
    sudo('mount -a')
    check_and_make_sym_link('/export/RTS', '/var/kat/archive/data/RTS')
    files.append('/etc/exports',
                    '/var/kat/data/nfs_staging 192.168.1.50(rw,sync,no_subtree_check)',
                    use_sudo=True)
    sudo('exportfs -a')

@task
@hosts(env.hosts)
def deploy():
    """Example usage: fab rts_dc.deploy"""
    #configure for proxy access
    site_proxy_configuration()

    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('umount /export/RTS')
        sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get update')

    # install ubuntu deb packages
    for pkg_list in DEB_PKGS: install_deb_packages(pkg_list)

    # pip install python packages
    for pkg in PIP_PKGS: install_pip_packages(pkg, flags='-U --no-deps')

    # install private ska-sa git packages
    for pkg in SKA_PRIVATE_GIT_PKGS: install_git_package(pkg, branch=GIT_BRANCH)

    # setup ntp
    ntp_configuration()

    # install oodt and related stuff
    make_directory_trees()
    auto_mounts()
    deploy_oodt()
    auto_start_filemgr()
    auto_start_crawler_rts()
