from fabric.api import sudo, task, hosts, settings, env
from fabric.contrib import files

from rts_common_deploy import install_deb_packages, install_pip_packages, install_git_package, retrieve_git_package
from rts_common_deploy import remove_deb_packages, retrieve_svn_package, install_svn_package, configure_and_make
from rts_common_deploy import make_directory, check_and_make_sym_link #, remove_dir
from rts_common_deploy import deploy_oodt_comp_ver_06 #, deploy_solr, configure_tomcat
from rts_common_deploy import OODT_HOME, OODT_CONF, VAR_KAT #, RTS_DATA, ARCHIVE_DATA, STAGING_HOME, STAGING_INGEST, STAGING_FAILED, SOLR_COLLECTIONS_HOME, 
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
OBIT_SVN_BASE = 'https://svn.cv.nrao.edu/svn/ObitInstall/'

#area to put katsdpscripts
SCRIPTS_AREA = '/var/kat/katsdpscripts'

#oodt flavoured directories specific to imager machine
WORKFLOW_AREA = '/var/kat/katsdpworkflow'

#log directories that need kat ownership
CELERY_LOG = '/var/log/celery'
TOMCAT7_LOG = '/var/log/tomcat7'
CAS_FILEMGR_LOG = '/var/log/cas_filemgr'

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
AIPS_TASKS = ['FILAIP','FITTP','LWPLA','POSSM','SNPLT','UVFLG','UVPLT']
AIPS_VERSION = '31DEC14'
AIPS_DIR = '/usr/local/aips'

def deploy_k7contpipe():
	"""
	Install the KAT-7 continuum pipeline and its dependencies.
	"""
	#install_svn_package('research/obit_imager')
	#deploy_obit()
	#Setup .katimrc
	#katimrc=['[KATPIPE]','aips_dir = ','aips_version = ','scratch_area = ','metadata_dir = ']
	#files.append('/home/kat/.katimrc',katimrc)
	#Get static data
	#retrieve_svn_package('research/obit_imager/FITS')
	#sudo('cp ')
	deploy_aips()

def deploy_obit():
	"""
	Checkout a skeletal form of a specific Obit revision which is just enough to get the
	kat-7 continuum pipeline to run.
	"""
	#Make a dir for obit
	sudo('mkdir ' + OBIT_INSTALL)
	#Extract the Obit repo from svn
	retrieve_svn_package('ObitSystem', base=OBIT_SVN_BASE, revision=OBIT_REVISION, output_location=OBIT_INSTALL)
	#Configure and make the base Obit package
	configure_and_make(OBIT_INSTALL + '/ObitSystem/Obit')
	#Copy the data from ObitTalk into Obits python setup
	sudo('cp -r' + OBIT_INSTALL + '/ObitSystem/ObitTalk/python ' + OBIT_INSTALL + '/ObitSystem/Obit/python')
	#Add Obits python module to sys.paths
	files.append('/usr/local/lib/python2.7/dist-packages/Obit.pth', OBIT_INSTALL + '/ObitSystem/ObitTalk/python', use_sudo=True)

def deploy_aips():
	"""
	Construct a minimal AIPS installation that can run the Obit pipeline.
	Use AIPSLite, to get package. AIPSLite should have been installed 
	"""
	try:
		from katim import AIPSLite
	except:
		raise ImportError('AIPSLite is not installed. Install the KAT-7 Obit pipeline first.')
	# Set up AIPS
	sudo('mkdir ' + AIPS_DIR)
	AIPSLite.get_aips(basedir=AIPS_DIR,version=AIPS_VERSION)
	# Download Packages
	AIPSLite.get_task(AIPS_TASKS)
	# AIPS needs environment variables set up in ~/.katimrc
	files.append('/home/kat/.katimrc','aips_dir = ' + AIPS_DIR)
	files.append('/home/kat/.katimrc','aips_version = ' + AIPS_VERSION)

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

    #auto-startup of filemgr
    check_and_make_sym_link('%s/%s' % (OODT_CONF, 'cas-filemgr/bin/cas-filemgr'), '/etc/init.d/cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc2.d/S93cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc3.d/S93cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc0.d/K07cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc6.d/K07cas-filemgr')
    sudo('/etc/init.d/cas-filemgr start')
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
