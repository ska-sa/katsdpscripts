#!/bin/bash
echo Checkout scripts
mkdir -p ~/scripts
cd ~/scripts
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnDS/code/ffinder/trunk/src/test/scripts .

mkdir -p ~/svn
[ "$FFSVN" ] || FFSVN="~/svn"
echo "Getting local configuration files"
mkdir -p /var/kat/conf
cd /var/kat/conf
svn co --username readonly --password svnds\ fooz3er https://katfs.kat.ac.za/svnDS/code/conf/ff .

echo "Getting katcp libraries"
mkdir -p $FFSVN/katcp
cd $FFSVN/katcp
svn co --username readonly --password svnds\ fooz3er https://katfs.kat.ac.za/svnDS/code/katcp-python/trunk .
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnROACH/sw/lib/katcp .

echo "Getting scape"
mkdir -p $FFSVN/scape
cd $FFSVN/scape
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnDS/code/scape/trunk .

echo "Getting katcore libraries"
mkdir -p $FFSVN/katcore
cd $FFSVN/katcore
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnDS/code/katcore/trunk .

echo "Getting katpoint library"
mkdir -p $FFSVN/katpoint
cd $FFSVN/katpoint
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnDS/code/katpoint/trunk .

echo "Getting katconf library"
mkdir -p $FFSVN/katconf
cd $FFSVN/katconf
svn co --username readonly --password svnds\ fooz3r https://katfs.kat.ac.za/svnDS/code/katconf/trunk .

echo "Getting fringe finder libraries"
cd $FFSVN/ffinder
mkdir -p $FFSVN/ffinder
svn co --username readonly --password svnds\ fooz3er https://katfs.kat.ac.za/svnDS/code/ffinder/trunk .

cd $FFSVN
ls -la
echo "Done. Press enter to exit"
read done
