from fabric.api import sudo, task, hosts, settings, env, run
from fabric.contrib import files
from fabric.context_managers import shell_env

from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package
from rts_common_deploy import remove_deb_packages, retrieve_svn_package, install_svn_package, configure_and_make
from rts_common_deploy import update_svn_package, make_directory, check_and_make_sym_link, rsync, remove_dir
from rts_common_deploy import auto_start_oodt_daemon
from rts_common_deploy import deploy_oodt_comp_ver_06 #, deploy_solr, configure_tomcat
from rts_common_deploy import OODT_HOME, OODT_CONF, VAR_KAT, ARCHIVE_HOME #, RTS_DATA, STAGING_HOME, STAGING_INGEST, STAGING_FAILED, SOLR_COLLECTIONS_HOME, 
from rts_common_deploy import GIT_BRANCH

#Set environment and hostnames
env.hosts = ['kat@192.168.6.185']
env.password = 'kat'

#rts-imager specific data areas
STAGING_AREA = '/data/staging_area'
PROCESS_AREA = '/data/process_area'

#Obit install location
OBIT_INSTALL = '/usr/local/Obit'
OBIT_REVISION = 483
OBIT_SVN_BASE = 'svn.cv.nrao.edu/svn/ObitInstall/'

#area to put katsdpscripts
SCRIPTS_AREA = '/var/kat/katsdpscripts'

#oodt flavoured directories specific to imager machine
WORKFLOW_AREA = '/var/kat/katsdpworkflow'

#log directories that need kat ownership
CELERY_LOG = '/var/log/celery'
TOMCAT7_LOG = '/var/log/tomcat7'
CAS_FILEMGR_LOG = '/var/log/cas_filemgr'
CAS_WOKKFLOWMGR_LOG = '/var/log/cas_workflowmgr'

# Deb packages for rts-imager
DEB_PKGS = [ 'vim', 'python-dev', 'gawk', 'pkg-config', 'libglib2.0-dev',
			 'libfftw3-dev', 'libgsl0-dev', 'libxmlrpc-core-c3-dev',            
			 'libcurl4-openssl-dev', 'libx11-dev', 'libice-dev', 'libcfitsio3-dev',  #general
             'gfortran', 'libatlas-base-dev', 'libblas-dev', 'libexpat1-dev',
             'git', 'git-man',                                                       #git
             'python-pip', 'python-setuptools', 'python-pkg-resources',              #pip
             'subversion', 'nfs-kernel-server', 'imagemagick',
             'python-celery', 'celeryd', 'rabbitmq-server',
             'tree', 'pyflakes', 'openjdk-7-jre', 'htop',                            #Other things
             'ipython', 'python-numpy', 'python-scipy', 'python-h5py',
             'python-matplotlib', 'python-pyfits', 'python-pandas', 'python-nose',   #python stuff
             'nfs-common']#'tomcat7', 'nfs-common']

# Pip packages for rts-imager
PIP_PKGS = ['pyephem', 'scikits.fitting', 'pysolr']

# SKA private git packages for rts imager
SKA_PRIVATE_GIT_PKGS = ['katpoint', 'katdal', 'katholog', 'scape', 'katsdpdata']

# AIPS Tasks to retrieve using AIPSLite
AIPS_TASKS = ['FITTP','LWPLA','POSSM','SNPLT','UVFLG','UVPLT','UVCOP']
AIPS_VERSION = '31DEC14'
AIPS_DIR = '/usr/local/aips'

def deploy_k7contpipe():
    """
    Install the KAT-7 continuum pipeline and its dependencies.
    """
    #Install obit_imager
    install_svn_package('obit_imager',repo='svnDS/research')
    #Setup .katimrc
    sudo('rm -f /home/kat/.katimrc')
    katimrc=['[KATPIPE]','aips_dir =','aips_version =','metadata_dir =','obit_dir =']
    files.append('/home/kat/.katimrc',katimrc)
    #Get static data and put it in /var/kat/k7contpipe
    sudo('mkdir -p /var/kat/k7contpipe')
    retrieve_svn_package('FITS', repo='svnDS/research/obit_imager',output_location='/var/kat/k7contpipe')
    files.sed('/home/kat/.katimrc','metadata_dir = *','metadata_dir = /var/kat/k7contpipe')
    # setup Obit
    deploy_obit()
    # setup AIPS
    deploy_aips()
	
def deploy_obit():
    """
    Checkout a skeletal form of a specific Obit revision which is just enough to get the
    kat-7 continuum pipeline to run.
    """
    #Make a dir for obit
    make_directory(OBIT_INSTALL)
    if files.exists(OBIT_INSTALL+'/.svn'):
        #Update the Obit repo  via svn.
        update_svn_package(OBIT_INSTALL, revision=OBIT_REVISION)
    else:
        #Extract the Obit repo from svn if not already in its final resting place
        retrieve_svn_package('ObitSystem', base=OBIT_SVN_BASE, repo='', revision=OBIT_REVISION, output_location=OBIT_INSTALL)
    #Configure and make the base Obit package
    configure_and_make(OBIT_INSTALL + '/Obit')
    #Copy the data from ObitTalk into Obits python setup
    sudo('cp -r ' + OBIT_INSTALL + '/ObitTalk/python ' + OBIT_INSTALL + '/Obit/python')
    #Add Obits python module to sys.paths
    files.append('/usr/local/lib/python2.7/dist-packages/Obit.pth', OBIT_INSTALL + '/Obit/python', use_sudo=True)
    #Set location of Obit install in .katimrc
    files.sed('/home/kat/.katimrc', 'obit_dir = *', 'obit_dir = '+OBIT_INSTALL+'/Obit')

def deploy_aips():
    """
    Construct a minimal AIPS installation that can run the Obit pipeline.
    This code is stolen from AIPSLite- and made to work with fabric 
    """

    #Delete old aips installation as this seems to conflict when updating
    remove_dir(AIPS_DIR)

    aips_server = 'ftp.aoc.nrao.edu'
    # Minimum files required:
    intel_libs = [AIPS_VERSION+'/LNX64/LIBR/INTELCMP/libimf.so', AIPS_VERSION+'/LNX64/LIBR/INTELCMP/libsvml.so']
    popsdat_files = [AIPS_VERSION+'/HELP/POPSDAT.HLP']
    binary_files = [AIPS_VERSION+'/LNX64/LOAD/FILAIP.EXE']
    
    make_directory(AIPS_DIR)
    # rsync the basic AIPS files
    rsync(aips_server, intel_libs+popsdat_files+binary_files, output_base=AIPS_DIR + '/' + AIPS_VERSION)
    #Sort out FILAIP
    data_dir = AIPS_DIR + '/' + AIPS_VERSION + '/DATA'
    mem_dir = AIPS_DIR + '/' + AIPS_VERSION + '/LNX64/MEMORY'
    template_dir = AIPS_DIR + '/' + AIPS_VERSION + '/LNX64/TEMPLATE'
    for temp_dir in [data_dir, mem_dir, template_dir]:
        make_directory(temp_dir)
    #Run FILAIP
    env={'DA00':template_dir, 'NET0':template_dir, 'DA01':data_dir, 'NVOL':'1', 'NEWMEM':mem_dir,
            'LD_LIBRARY_PATH':AIPS_DIR + '/' + AIPS_VERSION +'/LNX64/LIBR/INTELCMP/',
            'AIPS_VERSION':AIPS_DIR + '/' + AIPS_VERSION, 'AIPS_ROOT':AIPS_DIR,
            'VERSION':'NEW', 'NEW':AIPS_DIR + '/' + AIPS_VERSION}
    with(shell_env(**env)):
        run('echo 8 2 | ' + AIPS_DIR + '/' + AIPS_VERSION + '/LNX64/LOAD/FILAIP.EXE')
    # Download Tasks
    exe_files = [AIPS_VERSION + '/LNX64/LOAD/' + taskname + '.EXE' for taskname in AIPS_TASKS]
    hlp_files = [AIPS_VERSION + '/HELP/' + taskname + '.HLP' for taskname in AIPS_TASKS]
    rsync(aips_server, exe_files + hlp_files, output_base=AIPS_DIR + '/' + AIPS_VERSION)

    # AIPS needs environment variables set up in ~/.katimrc
    files.sed('/home/kat/.katimrc','aips_dir = *', 'aips_dir = ' + AIPS_DIR)
    files.sed('/home/kat/.katimrc','aips_version = *', 'aips_version = ' + AIPS_VERSION)

def deploy_oodt():
    deploy_oodt_comp_ver_06("cas-filemgr")

def make_directory_trees():
    make_directory(VAR_KAT)
    make_directory(ARCHIVE_HOME)
    make_directory(OODT_HOME)
    make_directory(OODT_CONF)
    make_directory(WORKFLOW_AREA)
    make_directory(STAGING_AREA)
    make_directory(PROCESS_AREA)
    make_directory(CELERY_LOG) #change owner
    make_directory(CAS_FILEMGR_LOG)
    make_directory(CAS_WOKKFLOWMGR_LOG)
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
    files.append('/usr/local/lib/python2.7/dist-packages/katsdpworkflow.pth', WORKFLOW_AREA, use_sudo=True)

    # retrieve katsdpscripts and install (need the RTS scripts in a locateable area)
    retrieve_git_package('katsdpscripts', output_location=SCRIPTS_AREA)
    install_pip_packages(SCRIPTS_AREA, flags='-U --no-deps')

    auto_start_oodt_daemon('cas-filemgr')
    #TODO make this deploy for our python interface XMLRPC interface for celery.
    #auto_start_workflow_rts()
    configure_matplotlib()
    configure_celery()
    deploy_k7contpipe()

