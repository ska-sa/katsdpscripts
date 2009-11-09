#!/bin/bash
echo Checkout scripts
[ "$FFhome" ] || FFhome="/home/ffuser"
mkdir -p $FFhome/svn
mkdir -p $FFhome/scripts
cd $FFhome/scripts
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/ffinder/trunk/src/test/scripts .

echo "Getting local configuration files"
mkdir -p /var/kat/conf
cd /var/kat/conf
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/conf/ff .

echo "Getting katcp libraries"
mkdir -p $FFhome/svn/katcp
cd $FFhome/svn/katcp
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/katcp-python/trunk .

echo "Getting scape"
mkdir -p $FFhome/svn/scape
cd $FFhome/svn/scape
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/scape/trunk .

echo "Getting katcore libraries"
mkdir -p $FFhome/svn/katcore
cd $FFhome/svn/katcore
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/katcore/trunk .

echo "Getting katpoint library"
mkdir -p $FFhome/svn/katpoint
cd $FFhome/svn/katpoint
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/katpoint/trunk .

echo "Getting katconf library"
mkdir -p $FFhome/svn/katconf
cd $FFhome/svn/katconf
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/katconf/trunk .

echo "Getting fringe finder libraries"
mkdir -p $FFhome/svn/ffinder
cd $FFhome/svn/ffinder
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnDS/code/ffinder/trunk .
mkdir -p $FFhome/svn/ffinder/roach
cd $FFhome/svn/ffinder/roach
svn co --username ffuser --password ffuser4svnup https://katfs.kat.ac.za/svnROACH/sw/lib/katcp .

cd $FFhome/svn
ls -la
echo "Done. Press enter to exit"
read done
