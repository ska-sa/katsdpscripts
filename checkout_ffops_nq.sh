#!/bin/bash
mkdir /home/ffuser/scripts
cd scripts
svn co https://katfs.kat.ac.za/svnDS/code/ffinder/trunk/src/test/scripts .

mkdir /home/ffuser/svn
cd svn
[ "$FFSVN" ] || FFSVN="/home/ffuser/svn"
echo "Getting local configuration files"
mkdir /var/kat/conf
cd /var/kat/conf
svn co https://katfs.kat.ac.za/svnDS/code/conf/ff .

echo "Getting katcp libraries"
mkdir $FFSVN/katcp
cd $FFSVN/katcp
svn co https://katfs.kat.ac.za/svnDS/code/katcp-python/trunk .
svn co https://katfs.kat.ac.za/svnROACH/sw/lib/katcp .

echo "Getting scape"
mkdir $FFSVN/scape
cd $FFSVN/scape
svn co https://katfs.kat.ac.za/svnDS/code/scape/trunk .

echo "Getting katcore libraries"
mkdir $FFSVN/katcore
cd $FFSVN/katcore
svn co https://katfs.kat.ac.za/svnDS/code/katcore/trunk .

echo "Getting katpoint library"
mkdir $FFSVN/katpoint
cd $FFSVN/katpoint
svn co https://katfs.kat.ac.za/svnDS/code/katpoint/trunk .

echo "Getting katconf library"
mkdir $FFSVN/katconf
cd $FFSVN/katconf
svn co https://katfs.kat.ac.za/svnDS/code/katconf/trunk .

echo "Getting fringe finder libraries"
cd $FFSVN/ffinder
mkdir $FFSVN/ffinder
svn co https://katfs.kat.ac.za/svnDS/code/ffinder/trunk .

cd $FFSVN
ls -la
echo "Done. Press enter to exit"
read done
