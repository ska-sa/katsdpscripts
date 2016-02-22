#!/bin/bash


pickle='../../RTS/rfi_mask.pickle'
noise='/var/kat/katconfig/user/noise-diode-models/mkat/'
pointing='/var/kat/katconfig/user/pointing-models/mkat'

# Identified by description: obs.sb.description = 'AR1: MKAIV-111 L-band Pointing L Band'
pointingfiles=("/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/17/1455726323.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/17/1455699514.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455651343.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455637289.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/12/1455298411.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455059941.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455045090.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455043359.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455041782.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/08/1454948408.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/05/1454691824.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/01/1454365764.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/01/1454323952.h5")
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/15/1455517819.h5"

len=${#pointingfiles[*]}
echo $len
for ((i=0; i < $len; i++))
do
    # create an ms file format
    echo
    echo ${pointingfiles[$i]}
    # use models from observation file (note: do not use the mc methods -- Lud does not trust it)
    ./analyse_point_source_scans.py -b --baseline=m062 -c $pickle ${pointingfiles[$i]}
    # overwrite models from observation file with user defined file
    # ./analyse_point_source_scans.py -b -a m062 -c $pickle -n $noise -p $pointing'/m062.pm.csv' ${pointingfiles[$i]}
done

# -fin-
