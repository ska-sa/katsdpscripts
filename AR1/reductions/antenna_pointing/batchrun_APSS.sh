#!/bin/bash

pickle='../../../RTS/rfi_mask.pickle'
noise='/var/kat/katconfig/user/noise-diode-models/mkat/'
pointing='/var/kat/katconfig/user/pointing-models/mkat'

##For all antennas in a single array
# Identified by description: obs.sb.description = 'AR1: MKAIV-111 L-band Pointing L Band'
pointingfiles=("/var/kat/archive2/data/MeerKATAR1/telescope_products/2016/06/16/1466110310.h5"
"/var/kat/archive2/data/MeerKATAR1/telescope_products/2016/06/17/1466136886.h5")

ant='m036'
len=${#pointingfiles[*]}
echo $len
for ((i=0; i < $len; i++))
do
    # create an ms file format
    echo
    echo ${pointingfiles[$i]}
    # use models from observation file (note: do not use the mc methods -- Lud does not trust it)
    ./analyse_point_source_scans.py -b --baseline=$ant -c $pickle ${pointingfiles[$i]}
    # overwrite models from observation file with user defined file
    # ./analyse_point_source_scans.py -b -a m062 -c $pickle -n $noise -p $pointing'/m062.pm.csv' ${pointingfiles[$i]}
done

# -fin-
