###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from collections import namedtuple, defaultdict

"""
Convenience class to contain a single sample.
*update_seconds* and *value_seconds* are in float seconds (UTC) since the Unix epoch
*status* may take the values {'unknown', 'nominal', 'warn', error', 'failure'}
*value* (represented as a string) may be a numeric, discrete or string type
"""


class Sample(namedtuple('Sample', 'update_seconds value_seconds status value')):
    __slots__ = ()

    def __new__(_cls, update_seconds, value_seconds, status, value):
        ''' Accept float or string values. Coerce the timestamps to be floats '''
        try:
            value = float(value)
        except ValueError:
            try:
                value = value.strip()
            except AttributeError:
                pass
        return tuple.__new__(_cls, (float(update_seconds), float(value_seconds), status.strip(), value))

    @classmethod
    def _make(_cls, iterable, new=tuple.__new__, len=len):
        result = new(_cls, iterable)
        if len(result) != 4:
            raise TypeError('Expected 4 arguments, got %d' % len(result))
        return result

    @classmethod
    def from_string(cls, raw_str):
        """ Alternative constructor. Expects a single comma-delimited string with the format:
        'update_seconds,value_seconds,status,value'
        """
        fields = raw_str.split(',', 3)
        self = Sample(*fields)
        return self

    @classmethod
    def parse(cls, response):
        """ Utility method to parse a multipart KATcp response and return a dict
        {sensor_name : list_of_samples} """
        # messages[0] is the reply, the rest are informs
        assert response.messages[0].arguments[0] == 'ok'
        result = defaultdict(list)
        for m in response.messages[1:]:
            sensor_name = m.arguments[0]
            csv_data = m.arguments[1].splitlines()
            result[sensor_name].extend([Sample.from_string(s) for s in csv_data])
        return result

    def __repr__(self):
        return "[%0.3f %0.3f %7s %s]" % (self.update_seconds, self.value_seconds, self.status, self.value)

    def __getstate__(self):
        return (self.update_seconds, self.value_seconds, self.status, self.value)

    def __setstate__(self, pickled):
        self = Sample(*pickled)

    @property
    def update_millis(self):
        return round(1000 * self.update_seconds)

    @property
    def value_millis(self):
        return round(1000 * self.value_seconds)
