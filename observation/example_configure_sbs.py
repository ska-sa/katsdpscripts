#!/usr/bin/env python
"""Observation user examples"""

import logging
from optparse import OptionParser
import signal
import sys
import time
from datetime import datetime, timedelta

import katconf
from katuilib import obsbuild, ScheduleBlockTypes, ScheduleBlockStates, ScheduleBlockOutcomes, ScheduleBlockPriorities

APP_NAME = 'katscripts'

def main():
    parser = OptionParser()
    parser.add_option('-c', '--config', dest='config',
            type="string", default="/var/kat/katconfig", metavar='CONF',
            help='look for configuration files in folder CONF (default=%default)' )
    parser.add_option('-s', '--system_resource', dest='system_resource',
            type="string", default='systems/local.conf', metavar='SYSTEM_RESOURCE',
            help='System resource file to load (default=%default)')
    parser.add_option('-l', '--logging', dest='logging',
            type="string", default=None, metavar='LOGGING',
            help='level to use for basic logging or name of logging configuration file; ' \
                   'default is /log/log.<SITENAME>.conf')
    (options, args) = parser.parse_args()

    # Setup configuration source
    katconf.set_config(katconf.environ(options.config))

    # set up Python logging
    katconf.configure_logging(options.logging)
    logfile = "kat.%s" % APP_NAME
    logger = logging.getLogger(logfile)

    #Continue trying to load the configuration
    while True:
        try:
            sysconf = katconf.SystemConfig(options.system_resource)
            #all ok continue
            break
        except Exception,err:
            logger.info("Waiting for configuration ... %s" % err)
            print "Waiting for configuration ... ",err
            time.sleep(3)

    sysconf = katconf.SystemConfig(options.system_resource)
    db_uri = sysconf.conf.get("katobs","db_uri")

    logger.info("Logging started")
    logger.info("Katobs examples before obsbuild: db_uri=%s" % db_uri)

    obs = obsbuild(user='test', db_uri=db_uri)
    print "===obs.status()==="
    obs.status()

    #Display the created sb
    sb_id_code = obs.sb.new(owner='test', antenna_spec='ant2,ant4', controlled_resource='dbe,rfe7')
    print "===NEW SB CREATED===", sb_id_code

    print "\n===obs.sb==="
    print obs.sb

    #Make some changes
    obs.sb.type = ScheduleBlockTypes.OBSERVATION
    today = time.strftime('%Y-%m-%d', time.gmtime())     # Get the current date string (GMT)
    obs.sb.desired_start_time = datetime.strptime(today+" 21:00:00", "%Y-%m-%d %H:%M:%S")
    obs.sb.expected_duration_seconds =  timedelta(minutes=30).seconds
    obs.sb.instruction_set = "%run basic_health_check.py"
    obs.sb.save()
    print "\n===obs.sb CHANGED==="
    print obs.sb

    #Make some resource changes
    obs.sb.antenna_spec = "all"
    obs.sb.controlled_resource='dbe7,rfe7'
    obs.sb.save()
    print "\n===obs.sb RESOURCES CHANGED==="
    print obs.sb

    #Revert some changes
    obs.sb.antenna_spec = "ant5,ant6"
    obs.sb.controlled_resource='dbe'
    obs.sb.reset()
    print "\n===obs.sb RESOURCES UNCHANGED==="
    print obs.sb

    print "\n===obs.status()==="
    obs.status()

    logger.info("Katobs examples completed.")

if __name__ == "__main__":
    main()
