from fabric.api import sudo, run, env, task, cd, settings
from fabric.contrib import files
import os

# debian package install info
# install packages grouped by main packages and their dependencies
DEB_PKGS = ['gcc binutils cpp cpp-4.7 gcc-4.7 libc-dev-bin libc6-dev libcloog-ppl1 libgcc-4.7-dev libgmp10 libgmpxx4ldbl libgomp1 libitm1 libmpc2 libmpfr4 libppl-c4 libppl12 libquadmath0 linux-libc-dev manpages-dev', # pyspead
        'build-essential dpkg-dev fakeroot g++ g++-4.7 libalgorithm-diff-perl libalgorithm-diff-xs-perl libalgorithm-merge-perl libdpkg-perl libfile-fcntllock-perl libstdc++6-4.7-dev make patch', # pyspead
        'git git-man liberror-perl', #git
        'subversion libapr1 libaprutil1 libneon27-gnutls libserf1 libsvn1', # svn
        'python-pip python-setuptools python-pkg-resources', # pip
        'python-dev libpython-dev python2.7-dev libpython2.7 libpython2.7-dev gfortran gfortran-4.7 libgfortran-4.7-dev libatlas-base-dev libatlas-dev libatlas3-base libblas-dev libblas3 libgfortran3 libexpat1-dev', #numpy, scipy
        'libpng12-dev libfreetype6-dev zlib1g-dev', # matplotlib
        'libhdf5-7 hdf5-helpers libhdf5-dev libjpeg-dev libjpeg-turbo8 libjpeg-turbo8-dev libjpeg8 libjpeg8-dev', # h5py
        'apache2-mpm-worker apache2-utils apache2.2-bin apache2.2-common libaprutil1-dbd-sqlite3 libaprutil1-ldap ssl-cert apache2 ', # apache2, for oodt
        'libgssglue1 libnfsidmap2 libtirpc1 rpcbind libevent-2.0-5 nfs-common nfs-kernel-server' # nfs-common and nfs-kernel-server, for oodt
        ]

# pip install info
PIP_PKGS = ['ipython',
            'numpy','scipy',
            'h5py', 'scikits.fitting',                            # katsdpingest
            'pyephem',                                            # katpoint
            'iniparse',                                           # cbf simulator
            'sqlalchemy',                                         # katcorelib
            'python-dateutil','pyparsing','tornado','nose',       # matplotlib
            'matplotlib','mplh5canvas'                            # katsdpdisp
            'ply','zope.interface','twisted','unittest2','mock',  # katcp
            'genshi', 'jsobject',                                 # katconf
            'blinker',                                            # katcore
            'enum', 'ansicolors', 'pytz',                         # katmisc
            'paramiko','pycrypto','ecdsa',                        # fabric
            'fabric','setuptools-git','cuisine',                  # katdeploy
            'pyfits']                                             # katholog

# git install info
GIT_DIR = '/home/kat/git'
GIT_USER = 'katpull'
GIT_PASSWORD = 'katpull4git'
PUBLIC_SKA_GIT = ['PySPEAD','katcp-python'] # can get pyephem and scikits.fitting from ska git too
PRIVATE_SKA_GIT = ['katpoint','katsdpingest','katdal','katholog','katsdpdisp','katsdpcontroller','katsdpscripts']

# svn install info
SVN_DIR = '/home/kat/svn'
SVN_USER = 'kat'
SVN_PASSWORD = 'kat'
KAT_SVN_BASE = 'katfs.kat.ac.za/svnDS/code'
SVN_PKGS = ['katconf','katcore','katlogger','katcorelib','katmisc','katdeploy'] # katoodt is svn installed separately

###################### Package lists for rts-imager ######################################
# Deb packages for rts-imager
IMAGER_DEB_PKGS = [ 'python-dev',                                                        #general
                    'gfortran libatlas-base-dev libblas-dev libexpat1-dev',              #numpy/scipy
                    'git git-man',                                                       #git
                    'python-pip python-setuptools python-pkg-resources',                 #pip
                    'libhdf5-dev',                                                       #h5py
                    'libpng12-dev libfreetype6-dev zlib1g-dev',                          #Matplotlib
                    'subversion', 'nfs-kernel-server',
                    'tree'
                    ]

# Pip packages for rts-imager
IMAGER_PIP_PKGS = ['ipython',
                   'numpy','scipy',
                   'h5py', 'scikits.fitting',
                   'matplotlib', 'pyfits']

# git for rts imager
IMAGER_SKA_GIT = ['katpoint','katdal','katholog','katsdpscripts']

#svn for rts imager
IMAGER_SVN_PKGS = ['scape','katarchive','katoodt']

# scripts svn install info - no longer needed as scripts are in github in katsdpscripts
#SCRIPT_SVN_DIR = '/home/kat/reduce'
#SCRIPT_SVN_BASE = 'katfs.kat.ac.za/svnScience/RTS'


# kat config info
CONFIG_DIR = '/var/kat/katconfig'
CONFIG_SVN_BASE = os.path.join(KAT_SVN_BASE, 'katconfig/trunk')

# dictionary of deploy names to package names, where they differ
PKG_DICT = {'PySPEAD':'spead','katcp-python':'katcp'}

# oodt install info
OODT_HOME = '/usr/local/oodt/'
STAGING_HOME = '/var/kat/data'
ARCHIVE_HOME = '/var/kat/archive'
SDP_MC = '/var/kat/sdpcontroller'
ARCHIVE_APACHE_HOME = '/var/www/archive'
# Subdirectories derived from roots
# Where we're going to put the oodt configuration files
OODT_CONF = os.path.join(OODT_HOME, "conf")
# Primary staging directory
SOLR_COLLECTIONS_HOME = os.path.join(ARCHIVE_HOME, "catalogs/solr")
STAGING_INGEST = os.path.join(STAGING_HOME, "staging")
# Where files go when ingest fails
STAGING_FAILED = os.path.join(STAGING_HOME, "failed")
# Secondary staging directory for remote archiving
STAGING_NFS_INGEST = os.path.join(STAGING_HOME, "nfs_staging")
# Archive directory where all the data products will be stored
ARCHIVE_DATA = os.path.join(ARCHIVE_HOME, "data")
# RTS storage directory where all the data products will be stored
RTS_DATA = os.path.join(ARCHIVE_DATA, "RTS")

#Solr and Tomcat
SOLR_VER = "4.4.0"
TOMCAT_VER = "7.0.42"
TOMCAT_CONF = os.path.join(OODT_HOME, 'apache-tomcat/conf/server.xml')

def install_deb_packages(packages, extra_apt_parameters='', use_sudo=True, with_yes=True):
    """Install debian packages listed in space-separated string"""
    print ' ---- Install debian packages ---- \n', packages, '\n'
    run_ = sudo if use_sudo else run
    if with_yes:
        run_('yes | DEBIAN_FRONTEND=noninteractive apt-get %s install %s' % (extra_apt_parameters, packages))
    else:
        # by default run without automated "yes" answers, as they should be unnecessary if all of the package
        # dependencies are explicitly installed
        run_('apt-get %s install %s' % (extra_apt_parameters, packages))

def remove_deb_packages(packages, extra_apt_parameters='', use_sudo=True):
    """Remove debian packages listed in space-separated string"""
    print ' ---- Remove debian packages ---- \n', packages, '\n'
    run_ = sudo if use_sudo else run
    run_('yes | DEBIAN_FRONTEND=noninteractive apt-get %s remove %s' % (extra_apt_parameters, packages))

def install_pip_packages(packages):
    """Pip install packages listed in space-separated string"""
    print ' ---- Install', packages, ' ---- \n'
    sudo('pip install %s' % (packages,))

def remove_pip_packages(packages):
    """Pip uninstall packages listed in space-separated string"""
    print ' ---- Uninstall', packages, ' ---- \n'
    # check if the deploy name is different to the package name
    try:
        if packages in PKG_DICT.keys():
            sudo('yes | pip uninstall %s' % (PKG_DICT[packages],))
        else:
            sudo('yes | pip uninstall %s' % (packages))
    except:
        # we don't get a nice named exception if the package isn't there
        print 'cannot uninstall \n'

def install_git_package(package, repo='ska-sa', **kwargs):
    """Clone and pip install git package"""
    print ' ---- Install', package, ' ---- \n'
    with cd(GIT_DIR):
        # clone the package
        try:
            # if the repository is private, user and password info must be provided
            user = kwargs['user']
            password = kwargs['password']
            run('git clone https://'+user+':'+password+'@github.com/'+repo+'/'+package+'.git')
        except KeyError:
            # if no user and password provided, assume the repository is public
            run('git clone https://@github.com/'+repo+'/'+package+'.git')
        # install the package
        sudo('pip install '+package+'/')

def update_git_package(package, repo='ska-sa', **kwargs):
    """Git update and pip update git package"""
    print ' ---- Update', package, ' ---- \n'
    full_git_dir = GIT_DIR+'/'+package
    with cd(full_git_dir):
        # clone the package
        try:
            # if the repository is private, user and password info must be provided
            user = kwargs['user']
            password = kwargs['password']
            run('git pull https://'+user+':'+password+'@github.com/'+repo+'/'+package+'.git')
        except KeyError:
            # if no user and password provided, assume the repository is public
            run('git pull https://@github.com/'+repo+'/'+package+'.git')
    with cd(GIT_DIR):
        # install the package
        sudo('pip install -U '+package+'/')

def install_svn_package(package, user, password, base=KAT_SVN_BASE):
    """Checkout and pip install svn package"""
    print ' ---- Install', package, ' ---- \n'
    with cd(SVN_DIR):
        # check out and install the package
        run('svn checkout https://'+base+'/'+package+'/trunk '+package+' --username='+user+' --password='+password)
        sudo('pip install '+package+'/')

def update_svn_package(package, user, password, base=KAT_SVN_BASE):
    """Svn update and pip update svn package"""
    print ' ---- Update', package, ' ---- \n'
    with cd(SVN_DIR):
        # check out and install the package
        run('svn up '+package+' --username='+user+' --password='+password)
        sudo('pip install -U '+package+'/')

def install_workflow_manager(user, password, base=KAT_SVN_BASE):
    """Get the workflow manager"""
    pass

def checkout_svn_files(base, loc, user, password, use_sudo=True):
    """Checkout files that don't need to be installed"""
    run_ = sudo if use_sudo else run
    print ' ---- Checkout svn files ---- \n'
    run_('svn checkout https://'+base+' '+loc+' --username='+user+' --password='+password)

def update_svn_files(loc, user, password, use_sudo=True):
    """Update files that don't need to be installed"""
    run_ = sudo if use_sudo else run
    print ' ---- Update svn files ---- \n'
    with cd(loc):
        run_('svn up --username='+user+' --password='+password)

def remove_dir(rmdir):
    sudo("rm -rf %s" % (rmdir,))

def make_oodt_home():
    sudo("if [[ ! -d /usr/local/oodt/ ]]; then mkdir %s; fi" % (OODT_HOME))
    sudo("chown -R %s %s" % (env.user, OODT_HOME))
    sudo("chgrp -R %s %s" % (env.user, OODT_HOME))

def check_and_make_directory(dir_to_make):
    sudo("if [[ ! -d %s ]]; then mkdir -p %s; fi" % (dir_to_make, dir_to_make))
    sudo("chown -R %s %s" % (env.user, dir_to_make))
    sudo("chgrp -R %s %s" % (env.user, dir_to_make))

def make_dc_directory_trees():
    #staging directories
    check_and_make_directory(STAGING_INGEST)
    check_and_make_directory(STAGING_NFS_INGEST)
    check_and_make_directory(STAGING_FAILED)
    check_and_make_directory(ARCHIVE_DATA)
    check_and_make_directory(SOLR_COLLECTIONS_HOME)
    check_and_make_directory(RTS_DATA)
    check_and_make_directory(SDP_MC)

def configure_tomcat():
    files.sed(TOMCAT_CONF,
              '<Connector port="8080" protocol="HTTP/1.1"',
              '<Connector port="8983" protocol="HTTP/1.1"',
              use_sudo=True,
              backup='.bak')

def deploy_oodt_comp(comp_to_install, comp_ver):
    comp_tar = "%s-dist.tar.gz" % (comp_ver)
    run("rm -rf %s" % (os.path.join(OODT_HOME, comp_to_install)))
    run("wget -O /tmp/%s http://kat-archive.kat.ac.za/oodt_installs/%s" % (comp_tar, comp_tar))
    run("tar xzvf /tmp/%s -C %s" % (comp_tar, OODT_HOME))
    run("mv %s %s" % (os.path.join(OODT_HOME, comp_ver), os.path.join(OODT_HOME, comp_to_install)))
    run("rm -f /tmp/%s" % (comp_tar))

def deploy_oodt_comp_ver_06(comp_to_install):
    deploy_oodt_comp(comp_to_install, "%s-0.6" % (comp_to_install))

def deploy_dc_oodt_comps():
    deploy_oodt_comp_ver_06("cas-filemgr")
    deploy_oodt_comp_ver_06("cas-crawler")
    deploy_tomcat()
    deploy_solr()

def deploy_oodt_conf(branch):
    """Deploy the oodt configuration for branch (i.e karoo/vkaroo/lab/comm VMs).

    Parameters
    ==========
    branch: The OODT configuration svn branch to checkout. OODT configuration can be
    found in '../svnDS/code/katconfig/{branch}/static/oodt/'. For example 'trunk' or
    'branches/karoo'.
    """
    run("rm -rf %s" % (OODT_CONF))
    run("svn co --username kat --password kat "
        "https://katfs.kat.ac.za/svnDS/code/katconfig/%s/static/oodt/ "
        "%s" % (branch, OODT_CONF))

def deploy_tomcat(comp_to_install="apache-tomcat"):
    deploy_oodt_comp(comp_to_install, "%s-%s" % (comp_to_install, TOMCAT_VER))

def deploy_solr(comp_to_install="solr"):
    deploy_oodt_comp(comp_to_install, "%s-%s" % (comp_to_install, SOLR_VER))

def deploy_solr_in_tomcat():
    run("cp %s/solr/dist/solr-%s.war %s/apache-tomcat/webapps/solr.war" % (OODT_HOME, SOLR_VER, OODT_HOME))
    run("cp %s/solr/example/lib/ext/* %s/apache-tomcat/lib" % (OODT_HOME, OODT_HOME))
    run("rsync -rv --progress %s/solr/ %s" % (OODT_CONF, SOLR_COLLECTIONS_HOME))
    run("rm -rf %s/solr " % (OODT_HOME))

def check_and_make_sym_link(L_src, L_dest):
    sudo("if [[ ! -L %s ]]; then ln -s %s %s; fi" % (L_dest, L_src, L_dest))

def export_dc_nfs_staging():
    files.append('/etc/exports',
                 '%s rts-imager.kat.ac.za(rw,sync,no_subtree_check)' % (STAGING_NFS_INGEST),
                 use_sudo=True)
    sudo('exportfs -a')

def oodt_setup():
    make_oodt_home()
    make_dc_directory_trees()
    deploy_dc_oodt_comps()
    deploy_oodt_conf('branches/oodt_0_6_upgrade')
    configure_tomcat()
    deploy_solr_in_tomcat()
    export_dc_nfs_staging()

def remove_oodt_directories():
    remove_dir(OODT_HOME)
    remove_dir(ARCHIVE_HOME)
    remove_dir(ARCHIVE_APACHE_HOME)
    remove_dir(STAGING_HOME)
    remove_dir(SDP_MC)

@task
def deploy():
    # update the apt-get database. Warn, rather than abort, if repos are missing
    with settings(warn_only=True):
        sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get update')

    #install deb packages: thin plooging
    # install ubuntu deb packages
    for pkg_list in IMAGER_DEB_PKGS: install_deb_packages(pkg_list)

    #install pip packages: thin plooging
    # pip install python packages
    for name in IMAGER_PIP_PKGS: install_pip_packages(name)

    # create git package directory
    check_and_make_directory(GIT_DIR)
    
    # Make svn directory
    check_and_make_directory(SVN_DIR)

    # install svn packages
    for name in IMAGER_SVN_PKGS: install_svn_package(name,SVN_USER,SVN_PASSWORD)

    # install public ska-sa git packages
    #for name in IMAGER_PUBLIC_SKA_GIT: install_git_package(name)

    # install private ska-sa git packages
    for name in IMAGER_SKA_GIT: install_git_package(name,user=GIT_USER,password=GIT_PASSWORD)

    # update katconfig files
    #print ' ---- Update katconfig ---- \n'
    #update_svn_files(CONFIG_DIR,SVN_USER,SVN_PASSWORD)
    
    #Create RTS scripts directory
    #check_and_make_directory(SCRIPT_SVN_DIR)

    # update RTS scripts from svnScience
    #print ' ---- Update RTS scripts ---- \n'
    #checkout_svn_files(SCRIPT_SVN_BASE,SCRIPT_SVN_DIR,SVN_USER,SVN_PASSWORD,False)

    # oodt setup and install
    oodt_setup()


@task
def update():
    # update git/svn packages to current master/trunk versions

    # update public ska-sa git packages
    for name in IMAGER_SKA_GIT: update_git_package(name)

    # update svn packages
    for name in IMAGER_SVN_PKGS: update_svn_package(name,SVN_USER,SVN_PASSWORD)

    # update RTS scripts from svnScience
    #print ' ---- Update RTS scripts ---- \n'
    #update_svn_files(SCRIPT_SVN_DIR,SVN_USER,SVN_PASSWORD,False)

    # still outstanding - oodt updates

@task
def clear():
    # remove oodt directories and packages
    remove_oodt_directories()
    remove_pip_packages('katoodt')

    # remove git and svn directories
    remove_dir(GIT_DIR)
    remove_dir(SVN_DIR)
    #remove_dir(SCRIPT_SVN_DIR)
    remove_dir(CONFIG_DIR)

    # remove ska-sa git packages
    for name in reversed(IMAGER_SKA_GIT): remove_pip_packages(name)

    # pip uninstall python packages
    for name in reversed(IMAGER_PIP_PKGS): remove_pip_packages(name)
    # hack to remove scikits, which is still hanging around
    sudo('rm -rf /usr/local/lib/python2.7/dist-packages/scikits')

    # remove ubuntu deb packages
    for pkg_list in reversed(IMAGER_DEB_PKGS): remove_deb_packages(pkg_list)

