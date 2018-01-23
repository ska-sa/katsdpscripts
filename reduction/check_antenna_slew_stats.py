#!/usr/bin/python
#
# This examines antenna requested and actual position values
# to determine Stats .
#
# Sean Passmoor
# 7 December 2017
#

import optparse
import pickle
import numpy as np
import katdal


def haversine(az1, el1, az2, el2):
    """
    Calculate the great circle distance between two points 
    (az1, el1)  to  (az2, el2) on a sphere.
    The inputs are in decimal degrees and the result
    is the seperation  of the two pairs 
    in decimal degrees. 
    """
    # convert decimal degrees to radians
    lon1 = np.radians(el1)
    lat1 = np.radians(az1)
    lon2 = np.radians(el2)
    lat2 = np.radians(az2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = np.degrees(1) # convert radians to decimal degrees 
    return c * r


def unpickle(h5, sensor_name):
    temp = np.zeros(h5.file[sensor_name][:].shape + (2,))
    for i, (timestamp, val_str) in enumerate(h5.file[sensor_name][:]):
        val = pickle.loads(val_str)
        temp[i, :] = timestamp, val
    return temp


def unpickle_cat(h5, sensor_name):
    cat_lookup = {}
    temp = np.zeros(h5.file[sensor_name][:].shape + (2,))
    for i, (timestamp, val_str) in enumerate(h5.file[sensor_name][:]):
        val = pickle.loads(val_str)
        if not cat_lookup.has_key(val):
            cat_lookup[val] = len(cat_lookup.keys()) + 1
        temp[i, :] = timestamp, cat_lookup[val]
    return temp, cat_lookup


def antenna_stats(h5, ants='', slew_from_angles=(30, 7)):
    import pandas as pd
    if ants == '':
        ants = [ant.name for ant in h5.ants]
    if type(ants) == str:
        ants = [ants]

    for ant in ants:
        request_az = unpickle(
            h5, 'TelescopeState/%s_pos_request_scan_azim' % (ant,))
        request_el = unpickle(
            h5, 'TelescopeState/%s_pos_request_scan_elev' % (ant,))
        az = unpickle(h5, 'TelescopeState/%s_pos_actual_scan_azim' % (ant,))
        el = unpickle(h5, 'TelescopeState/%s_pos_actual_scan_elev' % (ant,))
        activity, activity_lookup = unpickle_cat(
            h5, 'TelescopeState/%s_activity' % (ant,))

        ts_el = pd.Series(el[:, 1], index=(
            el[:, 0] * 1e9).astype('datetime64[ns]'))
        ts_request_el = pd.Series(request_el[:, 1], index=(
            request_el[:, 0] * 1e9).astype('datetime64[ns]')).reindex_like(ts_el, method='ffill')
        ts_az = pd.Series(az[:, 1], index=(
            az[:, 0] * 1e9).astype('datetime64[ns]'))
        ts_request_az = pd.Series(request_az[:, 1], index=(
            request_az[:, 0] * 1e9).astype('datetime64[ns]')).reindex_like(ts_az, method='ffill')
        ts_activity = pd.Series(activity[:, 1], index=(
            activity[:, 0] * 1e9).astype('datetime64[ns]'))
        #delta_el = ts_request_el-ts_el
        #delta_az = ts_request_az-ts_az
        delta_sky = haversine(
            ts_request_az.reindex_like(ts_az, method='nearest'),
            ts_request_el.reindex_like(ts_az, method='nearest'),
            ts_az.reindex_like(ts_az, method='nearest'),
            ts_el.reindex_like(ts_az, method='nearest'))

        locked = (delta_sky.rolling(pd.Timedelta(
            seconds=2), min_periods=1, ).max() < 0.01)
        locked.index = locked.index + \
            pd.Timedelta(seconds=2)  # move window foward edge
        locked = locked.reindex_like(
            ts_az, method='nearest') & (delta_sky < 0.01)

        if not activity_lookup.has_key('track'):
            activity_lookup['track'] = -1  # not found
        tracking = (ts_activity.reindex_like(ts_az, method='ffill')
                    == activity_lookup['track']) & (delta_sky < 0.01)
        if not activity_lookup.has_key('scan'):
            activity_lookup['scan'] = -1  # not found
        scaning = (ts_activity.reindex_like(ts_az, method='ffill')
                   == activity_lookup['scan']) & (delta_sky < 0.01)

        print "%s : Tracking Standard-Deviation = %8.2f arc-seconds" % (ant, delta_sky[tracking].std() * 3600)
        print "%s : Scanning Standard-Deviation = %8.2f arc-seconds" % (ant, delta_sky[scaning].std() * 3600)
        for angle_param in slew_from_angles:
            for i, x in enumerate((delta_sky > angle_param - 1.1) & (delta_sky < angle_param + 1.1)):
                if x:
                    lock_len = 60 + int(2 * angle_param)
                    # time to get lock_len
                    lock_len = np.min(
                        [delta_sky.shape[0] - i, lock_len]).astype(int)
                    for j in range(lock_len):
                        if delta_sky[i + j] < 0.01 and locked[i + j]:
                            seconds = (
                                delta_sky.iloc[[i + j]].index - delta_sky.iloc[[i]].index).total_seconds()[0]
                            az_diff = ts_az[i] - ts_az[i + j]
                            el_diff = ts_el[i] - ts_el[i + j]
                            distance = delta_sky[i] - delta_sky[i + j]
                            speed = distance / seconds
                            text = "%s :%s %4.2f deg slew-in in %4.2fs, avg-speed %4.3f deg/s at azel(%4.3f,%4.3f)  delta azel(%3.3f,%3.3f)"
                            print text % (ant, str(delta_sky.iloc[[i + j]].index[0])[:19], distance, seconds, speed, ts_az[i + j], ts_el[i + j], az_diff, el_diff)
                            break


parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description="This examines antenna requested and actual position values"
                               "in the given HDF5 file.")

(opts, args) = parser.parse_args()
if len(args) < 1:
    raise RuntimeError('Please specify an HDF5 file to check')

for filename in args:
    data = katdal.open(filename)
    antenna_stats(data)
