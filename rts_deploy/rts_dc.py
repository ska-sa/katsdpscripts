import os

from fabric.api import sudo, task, hosts, settings, env, run
from fabric.contrib import files

# from fabric.api import sudo, run, env, task, cd, settings
# from fabric.contrib import files
from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package
from rts_common_deploy import deploy_oodt_comp_ver_06, install_and_start_daemon, deploy_tarball
from rts_common_deploy import make_directory, check_and_make_sym_link
from rts_common_deploy import ntp_configuration
from rts_common_deploy import OODT_HOME, OODT_CONF
from rts_common_deploy import GIT_BRANCH

#Set environment and hostnames
env.hosts = ['kat@10.98.4.1']
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
             'nfs-common', 'tomcat7', 'ntp']

# Pip packages for rts-dc
PIP_PKGS = ['pyephem', 'scikits.fitting', 'pysolr', 'katcp', 'ProxyTypes']

# SKA private git packages for rts imager
SKA_PRIVATE_GIT_PKGS = ['katpoint', 'katdal', 'katholog', 'scape', 'PySPEAD', 'katsdpdata', 'katsdpcontroller', 'katsdpingest']

OODT_PKGS = ['cas-filemgr', 'cas-crawler']

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
ARCHIVE_MOUNT = '/export/RTS'

def install_solr(comp_to_install="solr"):
    SOLR_VER = "4.4.0"
    deploy_tarball(comp_to_install, "%s-%s" % (comp_to_install, SOLR_VER))
    sudo("cp %s/solr/dist/solr-%s.war /var/lib/tomcat7/webapps/solr.war" % (OODT_HOME, SOLR_VER,))
    sudo("cp %s/solr/example/lib/ext/* /usr/share/tomcat7/lib" % (OODT_HOME,))
    run("rsync -rv --progress %s/solr/ %s" % (OODT_CONF, SOLR_COLLECTIONS_HOME))
    run("rm -rf %s/solr " % (OODT_HOME))

def configure_tomcat():
    sudo('/etc/init.d/tomcat7 stop')
    files.sed('/etc/tomcat7/server.xml',
              '<Connector port="8080" protocol="HTTP/1.1"',
              '<Connector port="8983" protocol="HTTP/1.1"',
              use_sudo=True)
    files.append('/etc/tomcat7/tomcat-users.xml',
                 '<role rolename="manager-gui"/><user username="kat" password="kat" roles="manager-gui"/>',
                 use_sudo=True)
    files.sed('/etc/default/tomcat7',
              'TOMCAT7_USER=tomcat7',
              'TOMCAT7_USER=kat',
              use_sudo=True)
    files.sed('/etc/default/tomcat7',
              'TOMCAT7_GROUP=tomcat7',
              'TOMCAT7_GROUP=kat',
              use_sudo=True)
    files.append('/etc/default/tomcat7',
                 'CATALINA_OPTS="-Dsolr.solr.home=/var/kat/archive/catalogs/solr"', 
                 use_sudo=True)
    sudo('/etc/init.d/tomcat7 start')

def install_oodt_package(pkg):
    make_directory(OODT_HOME, options='')
    deploy_oodt_comp_ver_06(pkg)

def auto_mounts():
    make_directory(ARCHIVE_MOUNT, options='')
    make_directory(STAGING_NFS_INGEST)

    files.append('/etc/fstab',
                 'kat-archive.karoo.kat.ac.za:/mnt/md3000i/sci_proc/RTS ' + ARCHIVE_MOUNT + ' nfs  _netdev,rw,soft,intr,auto,tcp,bg 0 0',
                 use_sudo=True)
    sudo('mount -a')
    check_and_make_sym_link(ARCHIVE_MOUNT, '/var/kat/archive/data/RTS')
    files.append('/etc/exports',
                    STAGING_NFS_INGEST + ' 192.168.1.50(rw,sync,no_subtree_check)',
                    use_sudo=True)
    sudo('exportfs -a')

def protect_mounts():
    """Stop know services that access the archive and then unmount the archive NFS mount."""
    sudo('/etc/init.d/cas-crawler-rts stop')
    sudo('/etc/init.d/cas-filemgr stop')
    sudo('umount ' + ARCHIVE_MOUNT)

@task
@hosts(env.hosts)
def deploy():
    """Example usage: fab rts_dc.deploy

    Useful Info (maybe)
    -----------
    Linux distro expected: Ubuntu 10.04 LTS
    Disk partitioning /dev/sda1 == root partition

    Notes
    -----
    [TB - 3/11/14]: IP address for deployment is 10.98.4.1
    """
    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get update')

    # install ubuntu deb packages
    for pkg_list in DEB_PKGS: install_deb_packages(pkg_list)

    # pip install python packages
    for pkg in PIP_PKGS: install_pip_packages(pkg, flags='-U --no-deps')

    # install private ska-sa git packages
    for pkg in SKA_PRIVATE_GIT_PKGS: install_git_package(pkg, branch=GIT_BRANCH)

    # install oodt packages
    for pkg in OODT_PKGS: install_oodt_package(pkg)

    retrieve_git_package('oodt_conf', branch='rts_dc', output_location=OODT_CONF)

    install_solr()
    configure_tomcat()


    # setup ntp
    ntp_configuration()

    # install oodt and related stuff
    make_directory(VAR_KAT)
    make_directory(ARCHIVE_HOME)
    make_directory(SDP_MC)
    make_directory(SOLR_COLLECTIONS_HOME)
    make_directory(STAGING_INGEST)
    make_directory(STAGING_FAILED)
    make_directory(ARCHIVE_DATA)
    make_directory(CAS_FILEMGR_LOG)
    make_directory(CAS_CRAWLER_LOG)

    auto_mounts()

    install_and_start_daemon(os.path.join(OODT_CONF,'cas-filemgr/bin'), 'cas-filemgr')
    install_and_start_daemon(os.path.join(OODT_CONF,'cas-crawler-rts/bin'), 'cas-crawler-rts')

@task
@hosts(env.hosts)
# [TB] Left here for future use.
def testing():
    """Used for testing when updating deployment."""
    protect_mounts()
    deploy()
