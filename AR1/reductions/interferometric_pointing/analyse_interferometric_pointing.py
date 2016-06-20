#!/usr/bin/python
# Script that uses katsdpcal's calprocs to reduce data consisting of offset tracks on multiple point sources.
#

from katsdpcal import calprocs
import pickle
import katdal
import numpy as np
import scikits.fitting as fit
import katpoint
import optparse


def activity(h5,state = 'track'):
    """Activity Sensor because some of antennas have a mind of their own, 
    others appear to have lost theirs entirely """
    antlist = [a.name for a in h5.ants]
    activityV = np.zeros((len(antlist),h5.shape[0]) ,dtype=np.bool)
    for i,ant in enumerate(antlist) :
        sensor = h5.sensor['Antennas/%s/activity'%(ant)]
        activityV[i,:] +=   (sensor==state)
    return np.all(activityV,axis=0)

def defaulted_sensor(h5, quantity, default):
    """Safely get environmental sensor data. with defaults"""
    try:
        return h5.sensor[quantity] 
    except KeyError:
        return np.repeat(default, h5.shape[0])

def Ang_Separation(pos1,pos2):
    """Calculate the greatest circle distance between po1 and pos2[.....] in radians  """
    Ra1 = pos1[0]
    Dec1 = pos1[1]
    Ra2 = np.array(pos2[0,:])
    Dec2 = np.array(pos2[1,:])
    top = np.cos(Dec2)**2*np.sin(Ra2-Ra1)**2+(np.cos(Dec1)*np.sin(Dec2)-np.sin(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1))**2
    bottom = np.sin(Dec1)*np.sin(Dec2)+np.cos(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1)
    return np.arctan2(np.sqrt(top),(bottom))


def reduce_compscan_inf(h5 ,channel_mask = '/var/kat/katsdpscripts/RTS/rfi_mask.pickle',chunks=16,return_raw=False):
    """Break the band up into chunks"""
    chunk_size = chunks
    rfi_static_flags = np.tile(False, h5.shape[0])
    if len(channel_mask)>0:
        pickle_file = open(channel_mask)
        rfi_static_flags = pickle.load(pickle_file)
        pickle_file.close()
    gains_p = {}
    stdv = {}
    calibrated = False # placeholder for calibration
    if not return_raw:     # Calculate average target flux over entire band
        target = h5.catalogue.targets[h5.target_indices[0]]
        flux_spectrum = h5.catalogue.targets[h5.target_indices[0]].flux_density(h5.freqs) # include flags
        average_flux = np.mean([flux for flux in flux_spectrum if not np.isnan(flux)])
        temperature = np.mean(defaulted_sensor(h5, 'temperature', 35.0)(h5.timestamps[:]))
        pressure = np.mean(defaulted_sensor(h5, 'pressure', 950.0)(h5.timestamps[:]))
        humidity = np.mean(defaulted_sensor(h5, 'humidity', 15.0)(h5.timestamps[:]))
        wind_speed = np.mean(defaulted_sensor(h5, 'wind_speed', 0.0)(h5.timestamps[:]))
        wind_direction  = np.degrees(np.angle(np.mean(np.exp(1j*np.radians(defaulted_sensor(h5, 'wind_direction', 0.0)(h5.timestamps[:]))))) )# Vector Mean
        sun = katpoint.Target('Sun, special')
        # Calculate pointing offset
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        middle_time = np.median(h5.timestamps[:], axis=None)
        # work out the sun's angle
        sun_azel = katpoint.rad2deg(np.array(sun.azel(middle_time,antenna=h5.ants[0])))  
        #TODO  Sort out the sun angle for diff ants
        # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
        requested_azel = target.azel(middle_time)
        # Correct for refraction, which becomes the requested value at input of pointing model
        rc = katpoint.RefractionCorrection()
        requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
        requested_azel = katpoint.rad2deg(np.array(requested_azel))

    avg= np.zeros((chunk_size* 2,10,len(h5.ants)) ) # freq Chunks * pol , pos *8 , Ant     
    h5.antlist = [a.name for a in h5.ants]
    h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
    pols = ["H","V"] # Put in logic for Intensity
    for i,pol in enumerate(pols) :
        gains_p[pol] = []
        pos = []
        stdv[pol] = []
        h5.select(pol=pol)
        h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
        for scan in h5.scans() : 
            data = h5.vis[activity(h5,state = 'track')]
            if data.shape[0] > 1 :
                gains_p[pol].append(calprocs.g_fit(data[:,:,:].mean(axis=0),h5.bls_lookup,refant=0) )
                stdv[pol].append(np.ones((data.shape[0],data.shape[1],len(h5.ants))).sum(axis=0))#number of data points
                pos.append( [h5.az[:,:].mean(axis=0), h5.el[:,:].mean(axis=0)] ) # time,ant
        for ant in xrange(len(h5.ants)):
            for chunk in xrange(chunks):
                if np.array(pos).shape[0] > 1 : # a good proxy for data 
                    freq = slice(chunk*256,(chunk+1)*256)
                    rfi = ~rfi_static_flags[freq]   
                    fitobj  = fit.GaussianFit(np.array(pos)[:,:,ant].mean(axis=0),[1,1],1) 
                    x = np.column_stack((np.array(pos)[:,0,ant],np.array(pos)[:,1,ant]))
                    y = np.abs(np.array(gains_p[pol])[:,freq,:][:,rfi,ant]).mean(axis=1)
                    y_err = 1./np.sqrt(np.array(stdv[pol])[:,freq,:][:,rfi,ant].sum(axis=1))
                    gaussian = fitobj.fit(x.T,y,y_err )
                    avg[chunk+i*chunk_size,0:2,ant] = gaussian.mean
                    avg[chunk+i*chunk_size,2:4,ant] = gaussian.std_mean
                    avg[chunk+i*chunk_size,4:6,ant] = gaussian.std
                    avg[chunk+i*chunk_size,6:8,ant] = gaussian.std_std
                    avg[chunk+i*chunk_size,8,ant] = gaussian.height
                    avg[chunk+i*chunk_size,9,ant] = gaussian.std_height
    if return_raw :
        return avg
    else:
        ant_pointing = {}
        pols = ["HH","VV",'I']
        pol_ind = {}
        pol_ind['HH'] = np.arange(0.0*chunk_size,1.0*chunk_size,dtype=int)
        pol_ind['VV'] = np.arange(1.0*chunk_size,2.0*chunk_size,dtype=int) 
        pol_ind['I']  = np.arange(0.0*chunk_size,2.0*chunk_size,dtype=int) 
        if np.any(np.isfinite(np.average(avg[:,0:2,:],axis=0,weights=1./avg[:,2:4,:]**2)) ) : # a bit overboard
            for ant in xrange(len(h5.ants)):
                ant_pointing[h5.ants[ant].name] = {}
                ant_pointing[h5.ants[ant].name]["antenna"] = h5.ants[ant].name
                ant_pointing[h5.ants[ant].name]["dataset"] = h5.name.split('/')[-1].split('.')[0]
                ant_pointing[h5.ants[ant].name]["target"] = target.name
                ant_pointing[h5.ants[ant].name]["timestamp_ut"] =str(katpoint.Timestamp(middle_time))
                ant_pointing[h5.ants[ant].name]["data_unit"] = 'Jy' if calibrated else 'counts'
                ant_pointing[h5.ants[ant].name]["frequency"] = h5.freqs.mean()[
                ant_pointing[h5.ants[ant].name]["flux"] = average_flux
                ant_pointing[h5.ants[ant].name]["temperature"] =temperature
                ant_pointing[h5.ants[ant].name]["pressure"] =pressure
                ant_pointing[h5.ants[ant].name]["humidity"] =humidity
                ant_pointing[h5.ants[ant].name]["wind_speed"] =wind_speed
                ant_pointing[h5.ants[ant].name]["wind_direction"] =wind_direction
                ant_pointing[h5.ants[ant].name]["sun_az"] = sun_azel.tolist()[0]
                ant_pointing[h5.ants[ant].name]["sun_el"] = sun_azel.tolist()[1]
                ant_pointing[h5.ants[ant].name]["timestamp"] =middle_time.astype(int)
                ant_pointing[h5.ants[ant].name]["azimuth"] =np.average(avg[pol_ind["I"],0,ant],axis=0,weights=1./avg[pol_ind["I"],2,ant]**2)
                ant_pointing[h5.ants[ant].name]["elevation"] =np.average(avg[pol_ind["I"],1,ant],axis=0,weights=1./avg[pol_ind["I"],3,ant]**2)
                azel_beam = np.average(avg[pol_ind["I"],0:2,ant],axis=0,weights=1./avg[pol_ind["I"],2:4,ant]**2)
                # Make sure the offset is a small angle around 0 degrees
                offset_azel = katpoint.wrap_angle(azel_beam - requested_azel, 360.)
                ant_pointing[h5.ants[ant].name]["delta_azimuth"] =offset_azel.tolist()[0]
                ant_pointing[h5.ants[ant].name]["delta_elevation"] =offset_azel.tolist()[1]
                ant_pointing[h5.ants[ant].name]["delta_elevation_std"] =0.0#calc
                ant_pointing[h5.ants[ant].name]["delta_azimuth_std"] =0.0#calc
                for pol in pol_ind:
                    ant_pointing[h5.ants[ant].name]["beam_height_%s"%(pol)]     = np.average(avg[pol_ind[pol],8,ant],axis=0,weights=1./avg[pol_ind[pol],9,ant]**2)
                    ant_pointing[h5.ants[ant].name]["beam_height_%s_std"%(pol)] = np.sqrt(np.sum(1./avg[pol_ind[pol],9,ant]**2) )
                    ant_pointing[h5.ants[ant].name]["beam_width_%s"%(pol)]      = np.average(avg[pol_ind[pol],4:6,ant],axis=0,weights=1./avg[pol_ind[pol],6:8,ant]**2).mean() 
                    ant_pointing[h5.ants[ant].name]["beam_width_%s_std"%(pol)]  = np.sqrt(np.sum(1./avg[pol_ind[pol],6:8,ant]**2) )
                    ant_pointing[h5.ants[ant].name]["baseline_height_%s"%(pol)] = 0.0
                    ant_pointing[h5.ants[ant].name]["baseline_height_%s_std"%(pol)] = 0.0
                    ant_pointing[h5.ants[ant].name]["refined_%s"%(pol)] =  5.0  # I don't know what this means 
                    ant_pointing[h5.ants[ant].name]["azimuth_%s"%(pol)]       =np.average(avg[pol_ind[pol],0,ant],axis=0,weights=1./avg[pol_ind[pol],2,ant]**2)
                    ant_pointing[h5.ants[ant].name]["elevation_%s"%(pol)]     =np.average(avg[pol_ind[pol],1,ant],axis=0,weights=1./avg[pol_ind[pol],3,ant]**2)
                    ant_pointing[h5.ants[ant].name]["azimuth_%s_std"%(pol)]   =np.sqrt(np.sum(1./avg[pol_ind[pol],0,ant]**2) )
                    ant_pointing[h5.ants[ant].name]["elevation_%s_std"%(pol)] =np.sqrt(np.sum(1./avg[pol_ind[pol],1,ant]**2) )
        return ant_pointing



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it.")

parser.add_option("-c", "--channel-mask", default="/var/kat/katsdpscripts/RTS/rfi_mask.pickle", help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
parser.add_option("-o", "--output", dest="outfilebase",default=None,
                  help="Base name of output files (*.csv for output data and *.log for messages, "
                       "default is '<dataset_name>_interferometric_pointing')")

(opts, args) = parser.parse_args()

if len(args) != 1 or not args[0].endswith('.h5'):
    raise RuntimeError('Please specify a single HDF5 file as argument to the script')

channel_mask = opts.channel_mask #

output_fields = '%(dataset)s, %(target)s, %(timestamp_ut)s, %(azimuth).7f, %(elevation).7f, ' \
                '%(delta_azimuth).7f, %(delta_azimuth_std).7f, %(delta_elevation).7f, %(delta_elevation_std).7f, ' \
                '%(data_unit)s, %(beam_height_I).7f, %(beam_height_I_std).7f, %(beam_width_I).7f, ' \
                '%(beam_width_I_std).7f, %(baseline_height_I).7f, %(baseline_height_I_std).7f, %(refined_I).7f, ' \
                '%(beam_height_HH).7f, %(beam_width_HH).7f, %(baseline_height_HH).7f, %(refined_HH).7f, ' \
                '%(beam_height_VV).7f, %(beam_width_VV).7f, %(baseline_height_VV).7f, %(refined_VV).7f, ' \
                '%(frequency).7f, %(flux).4f, %(temperature).2f, %(pressure).2f, %(humidity).2f, %(wind_speed).2f, ' \
                '%(wind_direction).2f , %(sun_az).7f, %(sun_el).7f, %(timestamp)i \n'

output_field_names = [name.partition(')')[0] for name in output_fields[2:].split(', %(')]

h5 = katdal.open(args)  # THis is an old KAT-7 file with no fringestopping
h5.select(compscans='interferometric_pointing')
h5.antlist = [a.name for a in h5.ants]
h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
if opts.outfilebase is None :
    outfilebase =  "%s_%s"%(h5.name.split('/')[-1].split('.')[0], "interferometric_pointing")
else:
    outfilebase = opts.outfilebase
f = {}
for ant in xrange(len(h5.ants)):
    f[h5.ants[ant].name] = file('%s_%s.csv'%(outfilebase,h5.ants[ant].name), 'w')
    f[h5.ants[ant].name].write('# antenna = %s\n' % h5.ants[ant].description)
    f[h5.ants[ant].name].write(', '.join(output_field_names) + '\n')

for cscan in h5.compscans() :
    avg = reduce_compscan_inf(h5,channel_mask)
    print "Compound scan %i of field %s "%(cscan[0],cscan[2].name)
    if len(avg) > 0 : # if not an empty set
        for antname in avg:
            f[antname].write(output_fields % avg[antname]) 
      
for ant in xrange(len(h5.ants)):
    f[h5.ants[ant].name].close()
