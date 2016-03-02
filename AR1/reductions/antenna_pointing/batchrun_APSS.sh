#!/bin/bash


pickle='../../../RTS/rfi_mask.pickle'
noise='/var/kat/katconfig/user/noise-diode-models/mkat/'
pointing='/var/kat/katconfig/user/pointing-models/mkat'

# #m062
# # Identified by description: obs.sb.description = 'AR1: MKAIV-111 L-band Pointing L Band'
# pointingfiles=("/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/29/1456759234.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/27/1456557706.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/26/1456520379.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/26/1456505723.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456155415.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456149609.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456114745.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/17/1455726323.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455651343.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455637289.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/05/1454691824.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/01/1454365764.h5"
# "/var/kat/archive/data/MeerKATAR1/telescope_products/2016/03/01/1456853841.h5")

#m063
# Identified by description: obs.sb.description = 'AR1: MKAIV-111 L-band Pointing L Band'
pointingfiles=("/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/26/1456505723.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456155415.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456149609.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/22/1456114745.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/17/1455726323.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/17/1455699514.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455651343.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/16/1455637289.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455059941.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/09/1455045090.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/08/1454948408.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/05/1454691824.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/02/01/1454365764.h5"
"/var/kat/archive/data/MeerKATAR1/telescope_products/2016/03/01/1456853841.h5")


len=${#pointingfiles[*]}
echo $len
for ((i=0; i < $len; i++))
do
    # create an ms file format
    echo
    echo ${pointingfiles[$i]}
    # use models from observation file (note: do not use the mc methods -- Lud does not trust it)
    # ./analyse_point_source_scans.py -b --baseline=m062 -c $pickle ${pointingfiles[$i]}
    ./analyse_point_source_scans.py -b --baseline=m063 -c $pickle ${pointingfiles[$i]}
    # overwrite models from observation file with user defined file
    # ./analyse_point_source_scans.py -b -a m062 -c $pickle -n $noise -p $pointing'/m062.pm.csv' ${pointingfiles[$i]}
done

# -fin-
