#!/usr/bin/python

###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import logging
from collections import defaultdict
from katcp import BlockingClient, Message

class KatstoreClient(BlockingClient):

    def __init__(self, host, port, timeout = 15.0):
        super(KatstoreClient,self).__init__(host, port, timeout)
        self.timeout = timeout

    def __enter__(self):
        self.start(timeout=self.timeout)
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        self.join(timeout=self.timeout)

    def historical_sensor_data(self, sensor_names_filter, start_seconds, end_seconds,
                        period_seconds=-1, strategy='stepwise',
                        fetch_last_value=False, timeout=None):
        timeout = max(15, timeout or (end_seconds - start_seconds)/1000)
        reply, informs =  self.blocking_request(
            Message.request(
                'historical-sensor-data', sensor_names_filter,
                start_seconds, end_seconds, period_seconds,
                strategy, int(fetch_last_value), timeout),
            timeout = timeout
         )
        if reply.arguments[0] != 'ok' or int(reply.arguments[1]) == 0:
            return self._results(reply, None)
        else:
            return self._results(reply, informs)

    def _results(self, reply, informs):
        result_dict = defaultdict(list)
        if informs:
            for inform in informs:
                sensor_name, csv_data = inform.arguments
                data = result_dict[sensor_name]
                data.extend(csv_data.strip().split('\n'))
                result_dict[sensor_name] = data
        return result_dict

    def historical_sensor_list(self, sensor_filter=''):
        reply, informs = self.blocking_request(
                Message.request('historical-sensor-list', sensor_filter))
        if reply.arguments[0] == 'ok':
            result = [inform.arguments for inform in informs]
        else:
            logger.warn(reply)
            result = []
        return result

