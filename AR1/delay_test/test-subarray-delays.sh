#!/bin/bash

COUNT=1
SUB_NR=1
BAND="l"
PRODUCT="c856M4k"
INSTRUCTION_SET="run-obs-script /home/kat/AR1/observations/track.py"
ARGS=""
DESCRIPTION="Delay test script"
USE_DATA=1

while [[ $# > 1 ]]
do
key="$1"
case $key in
    -a|--ant)
    ANT="$2"
    shift # past argument
    ;;
    -s|--sub-nr)
    SUB_NR="$2"
    shift # past argument
    ;;
    -c|--count)
    COUNT="$2"
    shift # past argument
    ;;
    -b|--band)
    BAND="$2"
    shift # past argument
    ;;
    -p|--product)
    PRODUCT="$2"
    shift # past argument
    ;;
    -i|--instruction-set)
    INSTRUCTION_SET="$2"
    shift # past argument
    ;;
    -r|--args)
    ARGS="$2"
    shift # past argument
    ;;
    -d|--description)
    DESCRIPTION="$2"
    shift # past argument
    ;;
    -u|--use-data)
    USE_DATA="$2"
    shift # past argument
    ;;
    -h|--help|*)
    SHOW_HELP=true
    break
    ;;
esac
shift # past argument or value
done

if [ "$SHOW_HELP" = true ] || [ -z "$ANT" ] || [ -z "$SUB_NR" ] || [ -z "$BAND" ] || [ -z "$INSTRUCTION_SET" ]; then
    echo "Usage: test-subarray.sh [-a ant] [-s sub_nr] [-c repeat_count] [-b band] [-p product] [-i instruction_set] [-r args] [-u 0|1]"
    echo "       -a or --ant                            use the specified antenna in the subarray, eg. m011 (required parameter)"
    echo "       -s or --sub-nr                         use the specified subarray, eg. 4 will use subarray_4 (default 1)"
    echo "       -c or --count                          repeat the test this many times, eg. 5 will run the script 5 times (default = 1)"
    echo "       -b or --band                           specify the band for the subarray, eg. 'l' (default)"
    echo "       -p or --product                        specify the product for the subarray, eg. 'c856M4k' (default)"
    echo "       -i or --instruction-set                specify an instruction set for the SB to use, eg. 'run-obs-script ~/svn/katscripts/cam/basic_capture_start.py' (default)"
    echo "       -r or --args                           specify the instruction set's arguments, eg. '-t 30 -m 60' (default = '')"
    echo "       -d or --description                    specify the Schedule Block's description, eg. 'Basic script' (default)"
    echo "       -u or --use-data                       specify whether to use a data proxy, eg. 1 or 0 (default=1)"
    exit 1
fi

# export environment variables for the test_subarray_creation.py script to use
# there is currently no other easy way (except for plugins) to get cmd line variables through nosetests
export TESTING_ANT=$ANT
export TESTING_SUB_NR=$SUB_NR
export TESTING_BAND=$BAND
export TESTING_PRODUCT=$PRODUCT
export TESTING_INSTRUCTION_SET=$INSTRUCTION_SET
export TESTING_ARGS=$ARGS
export TESTING_DESCRIPTION=$DESCRIPTION
export TESTING_USE_DATA=$USE_DATA

echo
echo "***************Initialising test for antenna(s), $ANT, on subarray_$SUB_NR...***************"
echo

CMDsync="/home/kat/AR1/utilities/global_sync_AR1.py -o ruby --with-array --delayfile /home/kat/AR1/utilities/pps_delays.csv"
CMDobs="/home/kat/scripts/integration_tests/run_aqf.py /home/kat/AR1/delay_test/test_subarray_delays.py --quick"

for ((i=1; i<=$COUNT; i++)); do
    echo
    echo
    echo "---Script run count == $i---"
    echo
    echo "-Global Sync-"
    echo n | $CMDsync
    sleep 2
    echo "-Delay Obs-"
    $CMDobs
    sleep 2
    echo
    echo "---Script run finished, sleeping for 2 seconds before retrying the script...---"
done

echo
echo "***************Test for antenna(s), $ANT, on subarray_$SUB_NR completed.***************"
echo
