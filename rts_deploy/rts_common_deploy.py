from fabric.api import sudo, run, env
from fabric.contrib import files
import os

#GIT branch to use for deployment
GIT_BRANCH = 'master'

# oodt install info
VAR_KAT = '/var/kat'
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

def install_deb_packages(packages):
    """Install debian packages listed in space-separated string"""
    print ' ---- Install debian packages ---- \n', packages, '\n'
    sudo('apt-get -y install %s' % (packages))

def remove_deb_packages(packages):
    """Remove debian packages listed in space-separated string"""
    print ' ---- Remove debian packages ---- \n', packages, '\n'
    sudo('apt-get -y remove %s' % (packages))

def install_pip_packages(packages, flags='-U'):
    """Pip install packages listed in space-separated string"""
    print ' ---- Install', packages, ' ---- \n'
    sudo('pip install '+flags+' %s' % (packages,))

def remove_pip_packages(packages):
    """Pip uninstall packages listed in space-separated string"""
    print ' ---- Uninstall', packages, ' ---- \n'
    try:
        sudo('yes | pip uninstall %s' % (packages))
    except:
        # we don't get a nice named exception if the package isn't there
        print 'Cannot uninstall \n'

def install_git_package(package, repo='ska-sa', login='katpull:katpull4git', branch='master', flags='-I --no-deps',**kwargs):
    """Install git packages directly using pip"""
    print ' ---- Install', package, ' ---- \n'
    if login:
        sudo('pip install '+ flags +' git+https://'+login+'@github.com/'+repo+'/'+package+'.git'+'@'+branch+'#egg='+package)
    else:
        sudo('pip install '+ flags +' git+https://github.com/'+repo+'/'+package+'.git@'+branch+'#egg='+package)

def retrieve_git_package(package, output_location=None, repo='ska-sa', login='katpull:katpull4git', branch='master', flags=''):
    """Copy a github repository to a specific location,
    overwriting contents of the output directory"""
    # Default package output location is the package name in the current directory
    if output_location is None:
        output_location = os.path.join(os.path.curdir,package)
    # Remove output location
    sudo('rm -rf '+output_location)
    print '\n ---- Retrieve', package, 'to', output_location, ' ---- \n'
    if login:
        run('git clone '+flags+' --branch '+branch+' https://'+login+'@github.com/'+repo+'/'+package+' '+output_location)
    else:
        run('git clone '+flags+' --branch '+branch+' https://github.com/'+repo+'/'+package+' '+output_location)

def remove_dir(rmdir):
    sudo("rm -rf %s" % (rmdir,))

def make_directory(dir_to_make):
    sudo("mkdir -p %s" % (dir_to_make))
    sudo("chown -R %s %s" % (env.user, dir_to_make))
    sudo("chgrp -R %s %s" % (env.user, dir_to_make))

def make_dc_directory_trees():
    #staging directories
    make_directory(STAGING_INGEST)
    make_directory(STAGING_NFS_INGEST)
    make_directory(STAGING_FAILED)
    make_directory(ARCHIVE_DATA)
    make_directory(SOLR_COLLECTIONS_HOME)
    make_directory(RTS_DATA)
    make_directory(SDP_MC)

def deploy_tarball(comp_to_install, comp_ver):
    comp_tar = "%s-dist.tar.gz" % (comp_ver)
    run("rm -rf %s" % (os.path.join(OODT_HOME, comp_to_install)))
    run("wget -O /tmp/%s http://kat-archive.kat.ac.za/oodt_installs/%s" % (comp_tar, comp_tar))
    run("tar xzvf /tmp/%s -C %s" % (comp_tar, OODT_HOME))
    run("mv %s %s" % (os.path.join(OODT_HOME, comp_ver), os.path.join(OODT_HOME, comp_to_install)))
    run("rm -f /tmp/%s" % (comp_tar))

def deploy_oodt_comp_ver_06(comp_to_install):
    deploy_tarball(comp_to_install, "%s-0.6" % (comp_to_install))

def check_and_make_sym_link(L_src, L_dest):
    sudo("if [[ ! -L %s ]]; then ln -s %s %s; fi" % (L_dest, L_src, L_dest))

def export_dc_nfs_staging():
    files.append('/etc/exports',
                 '%s rts-imager.kat.ac.za(rw,sync,no_subtree_check)' % (STAGING_NFS_INGEST),
                 use_sudo=True)
    sudo('exportfs -a')

def remove_oodt_directories():
    remove_dir(OODT_HOME)
    remove_dir(ARCHIVE_HOME)
    remove_dir(ARCHIVE_APACHE_HOME)
    remove_dir(STAGING_HOME)
    remove_dir(SDP_MC)

def deploy_solr(comp_to_install="solr"):
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

#auto-startup of filemgr
def auto_start_filemgr():
    check_and_make_sym_link('%s/%s' % (OODT_CONF, 'cas-filemgr/bin/cas-filemgr'), '/etc/init.d/cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc2.d/S93cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc3.d/S93cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc0.d/K07cas-filemgr')
    check_and_make_sym_link('/etc/init.d/cas-filemgr', '/etc/rc6.d/K07cas-filemgr')
    sudo('/etc/init.d/cas-filemgr start')

def auto_start_crawler_rts():
    check_and_make_sym_link('%s/%s' % (OODT_CONF, 'cas-crawler-rts/bin/cas-crawler-rts'), '/etc/init.d/cas-crawler-rts')
    check_and_make_sym_link('/etc/init.d/cas-crawler-rts', '/etc/rc2.d/S94cas-crawler-rts')
    check_and_make_sym_link('/etc/init.d/cas-crawler-rts', '/etc/rc3.d/S94cas-crawler-rts')
    check_and_make_sym_link('/etc/init.d/cas-crawler-rts', '/etc/rc0.d/K08cas-crawler-rts')
    check_and_make_sym_link('/etc/init.d/cas-crawler-rts', '/etc/rc6.d/K08cas-crawler-rts')
    sudo('/etc/init.d/cas-filemgr start')

def site_proxy_configuration():
    files.append('/etc/profile',
                    ['','#temporary proxy settings',
                    'export https_proxy=http://proxy.kat.ac.za:3128',
                    'export http_proxy=http://proxy.kat.ac.za:3128',
                    'export HTTPS_PROXY=http://proxy.kat.ac.za:3128',
                    'export HTTP_PROXY=http://proxy.kat.ac.za:3128'],
                    use_sudo=True)
    files.append('/etc/apt/apt.conf',
                    ['#temporary proxy settings',
                    'Acquire::http::Proxy "http://proxy.kat.ac.za:3128/";'],
                    use_sudo=True)

