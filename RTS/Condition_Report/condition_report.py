#!/usr/bin/python
#
#Produce a report on the weather conditions during an h5 file.
#Plot weather conditions and label the time ranges in the file where
#conditions are 'ideal','optimal' and 'normal'
#
#Conditions:
#'ideal':
#a) Night time only (sun elevation <-5deg.)
#b) Ambient temperature >= 19degC and <= 21degC 
#c) Ambient temperature rate of change <= 1degC in 30min
#d) No wind (wind speed < 1 m/s always)
#e) No precipitation
#Optimal:
#Optimal operating conditions are defined as follows:
#a) Night time only
#b) Ambient temperature >= -5degC and <= 35degC 
#c) Ambient temperature rate of change <= 2degC in 10min
#d) sustained 5-minute mean wind speed <= 2.9 m/s
#e) 3-second wind gust <= 4.1 m/s  (test to be done over 5 seconds by default- due to averaging)
#e) No precipitation
#Normal:
#Normal operating conditions are defined as follows:
#a) Day or Night
#b) Ambient temperature >= -5degC and <= 40degC 
#c) Ambient temperature rate of change <= 3degC in 20min
#d) sustained 5-minute mean wind speed <= 9.8 m/s
#e) 3-second wind gust <= 13.4 m/s
#f) Rain at a rate of 10mm/hour
#g) No hail, ice or snow

import optparse
from katsdpscripts.RTS.condition_report import condition_report

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-o", "--outputdir", type="string",default=".",help="Directory to save output file")
    parser.add_option("-a","--average",type="float",default=5.0,help="Averaging time in seconds for weather data. (default=5sec)")
    return parser.parse_args()

opts, args = parse_arguments()

condition_report(args[0], output_dirname=opts.outputdir, average_time=opts.average)


