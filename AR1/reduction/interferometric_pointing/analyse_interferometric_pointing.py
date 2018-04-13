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

#TODO Remove this function once katdal has this functionality 
def activity(h5,state = 'track'):
    """Activity Sensor because some of antennas have a mind of their own, 
    others appear to have lost theirs entirely """
    antlist = [a.name for a in h5.ants]
    activityV = np.zeros((len(antlist),h5.shape[0]) ,dtype=np.bool)
    for i,ant in enumerate(antlist) :
        sensor = h5.sensor['Antennas/%s/activity'%(ant)]
        activityV[i,:] +=   (sensor==state)
    return np.all(activityV,axis=0)

def w_average(arr,axis=None, weights=None):
    return np.nansum(arr*weights,axis=axis)/np.nansum(weights,axis=axis)

def reduce_compscan_inf(h5 ,channel_mask = None,chunks=16,return_raw=False):
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
        temperature = np.mean(h5.temperature)
        pressure = np.mean(h5.pressure)
        humidity = np.mean(h5.humidity)
        wind_speed = np.mean(h5.wind_speed)
        wind_direction  = np.degrees(np.angle(np.mean(np.exp(1j*np.radians(h5.wind_direction)))) )# Vector Mean
        sun = katpoint.Target('Sun, special')
        # Calculate pointing offset
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        middle_time = np.median(h5.timestamps[:], axis=None)
        # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
        requested_azel = target.azel(middle_time)
        # Correct for refraction, which becomes the requested value at input of pointing model
        rc = katpoint.RefractionCorrection()
        requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
        requested_azel = katpoint.rad2deg(np.array(requested_azel))

   
    gaussian_centre     = np.zeros((chunk_size* 2,2,len(h5.ants)) )
    gaussian_centre_std = np.zeros((chunk_size* 2,2,len(h5.ants)) )
    gaussian_width      = np.zeros((chunk_size* 2,2,len(h5.ants)) )
    gaussian_width_std  = np.zeros((chunk_size* 2,2,len(h5.ants)) )
    gaussian_height     = np.zeros((chunk_size* 2,len(h5.ants)) )
    gaussian_height_std = np.zeros((chunk_size* 2,len(h5.ants)) )
    pols = ["H","V"] # Put in logic for Intensity
    for i,pol in enumerate(pols) :
        gains_p[pol] = []
        pos = []
        stdv[pol] = []
        h5.select(pol=pol,corrprods='cross',ants=h5.antlist)
        h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
        for scan in h5.scans() : 
            valid_index = activity(h5,state = 'track')
            data = h5.vis[valid_index]
            if data.shape[0] > 0 : # need at least one data point
                gains_p[pol].append(calprocs.g_fit(data[:,:,:].mean(axis=0),h5.bls_lookup,refant=0) )
                stdv[pol].append(np.ones((data.shape[0],data.shape[1],len(h5.ants))).sum(axis=0))#number of data points
                # Get coords in (x(time,ants),y(time,ants) coords) 
                pos.append( [h5.target_x[valid_index,:].mean(axis=0), h5.target_y[valid_index,:].mean(axis=0)] ) 
        for ant in range(len(h5.ants)):
            for chunk in range(chunks):
                if np.array(pos).shape[0] > 4 : # Make sure there is enough data for a fit
                    freq = slice(chunk*(h5.shape[1]//chunks),(chunk+1)*(h5.shape[1]//chunks))
                    rfi = ~rfi_static_flags[freq]   
                    fitobj  = fit.GaussianFit(np.array(pos)[:,:,ant].mean(axis=0),[1.,1.],1)
                    x = np.column_stack((np.array(pos)[:,0,ant],np.array(pos)[:,1,ant]))
                    y = np.abs(np.array(gains_p[pol])[:,freq,:][:,rfi,ant]).mean(axis=1)
                    y_err = 1./np.sqrt(np.array(stdv[pol])[:,freq,:][:,rfi,ant].sum(axis=1))
                    gaussian = fitobj.fit(x.T,y,y_err ) 
                    #Fitted beam center is in (x, y) coordinates, in projection centred on target
                    snr = np.abs(np.r_[gaussian.std/gaussian.std_std])
                    valid_fit = np.all(np.isfinite(np.r_[gaussian.mean,gaussian.std_mean,gaussian.std,gaussian.std_std,gaussian.height,gaussian.std_height,snr]))
                    theta =  np.sqrt((gaussian.mean**2).sum())  # this is to see if the co-ord is out of range
                    #The valid fit is needed because I have no way of working out if the gain solution was ok.
                    if  not valid_fit or np.any(theta > np.pi) : # the checks to see if the fit is ok
                        gaussian_centre[chunk+i*chunk_size,:,ant]     =  np.nan
                        gaussian_centre_std[chunk+i*chunk_size,:,ant] =  np.nan
                        gaussian_width[chunk+i*chunk_size,:,ant]      =  np.nan
                        gaussian_width_std[chunk+i*chunk_size,:,ant]  =  np.nan
                        gaussian_height[chunk+i*chunk_size,ant]       =  np.nan
                        gaussian_height_std[chunk+i*chunk_size,ant]   =  np.nan
                    else:
                        # Convert this offset back to spherical (az, el) coordinates
                        beam_center_azel = target.plane_to_sphere(np.radians(gaussian.mean[0]), np.radians(gaussian.mean[1]), middle_time)
                        # Now correct the measured (az, el) for refraction and then apply the old pointing model
                        # to get a "raw" measured (az, el) at the output of the pointing model
                        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
                        beam_center_azel = h5.ants[ant].pointing_model.apply(*beam_center_azel)
                        beam_center_azel = np.degrees(np.array(beam_center_azel))
                        gaussian_centre[chunk+i*chunk_size,:,ant]     = beam_center_azel
                        gaussian_centre_std[chunk+i*chunk_size,:,ant] = gaussian.std_mean
                        gaussian_width[chunk+i*chunk_size,:,ant]      = gaussian.std
                        gaussian_width_std[chunk+i*chunk_size,:,ant]  = gaussian.std_std
                        gaussian_height[chunk+i*chunk_size,ant]       = gaussian.height
                        gaussian_height_std[chunk+i*chunk_size,ant]   = gaussian.std_height

    if return_raw :
        return np.r_[gaussian_centre , gaussian_centre_std , gaussian_width , gaussian_width_std , gaussian_height , gaussian_height_std]
    else:
        ant_pointing = {}
        pols = ["HH","VV",'I']
        pol_ind = {}
        pol_ind['HH'] = np.arange(0.0*chunk_size,1.0*chunk_size,dtype=int)
        pol_ind['VV'] = np.arange(1.0*chunk_size,2.0*chunk_size,dtype=int) 
        pol_ind['I']  = np.arange(0.0*chunk_size,2.0*chunk_size,dtype=int) 
        for ant in range(len(h5.ants)):
            if np.any(np.isfinite(w_average(gaussian_centre[:,:,ant],axis=0,weights=1./gaussian_centre_std[:,:,ant]**2)) ) : # a bit overboard
                name = h5.ants[ant].name
                ant_pointing[name] = {}
                ant_pointing[name]["antenna"] = h5.ants[ant].name
                ant_pointing[name]["dataset"] = h5.name.split('/')[-1].split('.')[0]
                ant_pointing[name]["target"] = target.name
                ant_pointing[name]["timestamp_ut"] =str(katpoint.Timestamp(middle_time))
                ant_pointing[name]["data_unit"] = 'Jy' if calibrated else 'counts'
                ant_pointing[name]["frequency"] = h5.freqs.mean()
                ant_pointing[name]["flux"] = average_flux
                ant_pointing[name]["temperature"] =temperature
                ant_pointing[name]["pressure"] =pressure
                ant_pointing[name]["humidity"] =humidity
                ant_pointing[name]["wind_speed"] =wind_speed
                ant_pointing[name]["wind_direction"] =wind_direction
                # work out the sun's angle
                sun_azel = katpoint.rad2deg(np.array(sun.azel(middle_time,antenna=h5.ants[ant])))  
                ant_pointing[name]["sun_az"] = sun_azel.tolist()[0]
                ant_pointing[name]["sun_el"] = sun_azel.tolist()[1]
                ant_pointing[name]["timestamp"] =middle_time.astype(int)
                #Work out the Target position and the requested position
                # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
                requested_azel = target.azel(middle_time,antenna=h5.ants[ant])
                # Correct for refraction, which becomes the requested value at input of pointing model
                rc = katpoint.RefractionCorrection()
                requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
                requested_azel = katpoint.rad2deg(np.array(requested_azel))
                target_azel = katpoint.rad2deg(np.array(target.azel(middle_time,antenna=h5.ants[ant])))  
                ant_pointing[name]["azimuth"] =target_azel.tolist()[0]
                ant_pointing[name]["elevation"] =target_azel.tolist()[1]
                azel_beam = w_average(gaussian_centre[pol_ind["I"],:,ant],axis=0,weights=1./gaussian_centre_std[pol_ind["I"],:,ant]**2)
                # Make sure the offset is a small angle around 0 degrees
                offset_azel = katpoint.wrap_angle(azel_beam - requested_azel, 360.)
                ant_pointing[name]["delta_azimuth"] =offset_azel.tolist()[0]
                ant_pointing[name]["delta_elevation"] =offset_azel.tolist()[1]
                ant_pointing[name]["delta_elevation_std"] =0.0#calc
                ant_pointing[name]["delta_azimuth_std"] =0.0#calc
                for pol in pol_ind:
                    ant_pointing[name]["beam_height_%s"%(pol)]     = w_average(gaussian_height[pol_ind[pol],ant],axis=0,weights=1./gaussian_height_std[pol_ind[pol],ant]**2)
                    ant_pointing[name]["beam_height_%s_std"%(pol)] = np.sqrt(np.nansum(1./gaussian_height_std[pol_ind[pol],ant]**2) )
                    ant_pointing[name]["beam_width_%s"%(pol)]      = w_average(gaussian_width[pol_ind[pol],:,ant],axis=0,weights=1./gaussian_width_std[pol_ind[pol],:,ant]**2).mean() 
                    ant_pointing[name]["beam_width_%s_std"%(pol)]  = np.sqrt(np.nansum(1./gaussian_width_std[pol_ind[pol],:,ant]**2) )
                    ant_pointing[name]["baseline_height_%s"%(pol)] = 0.0
                    ant_pointing[name]["baseline_height_%s_std"%(pol)] = 0.0
                    ant_pointing[name]["refined_%s"%(pol)] =  5.0  # I don't know what this means 
                    ant_pointing[name]["azimuth_%s"%(pol)]       =w_average(gaussian_centre[pol_ind[pol],0,ant],axis=0,weights=1./gaussian_centre_std[pol_ind[pol],0,ant]**2)
                    ant_pointing[name]["elevation_%s"%(pol)]     =w_average(gaussian_centre[pol_ind[pol],1,ant],axis=0,weights=1./gaussian_centre_std[pol_ind[pol],1,ant]**2)
                    ant_pointing[name]["azimuth_%s_std"%(pol)]   =np.sqrt(np.nansum(1./gaussian_centre_std[pol_ind[pol],0,ant]**2) )
                    ant_pointing[name]["elevation_%s_std"%(pol)] =np.sqrt(np.nansum(1./gaussian_centre_std[pol_ind[pol],1,ant]**2) )

        return ant_pointing



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it.")
parser.add_option("-a", "--ants", dest="ants",default=None,
                  help="List of antennas to use in the reduction "
                       "default is all antennas in the data set")
parser.add_option( "--exclude-ants", dest="ex_ants",default=None,
                  help="List of antennas to exculde from the reduction "
                       "default is None of the antennas in the data set")

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

h5 = katdal.open(args)  
ant_list = [ant.name for ant in h5.ants] # Temp list for input options
if opts.ants is not None  :
    ant_list = opts.ants.split(',')
if opts.ex_ants is not None :
    for ant in opts.ex_ants.split(','):
        if ant in ant_list:
            ant_list.remove(ant)
h5.select(compscans='interferometric_pointing',ants=ant_list)

h5.antlist = [a.name for a in h5.ants]
h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
if opts.outfilebase is None :
    outfilebase =  "%s_%s"%(h5.name.split('/')[-1].split('.')[0], "interferometric_pointing")
else:
    outfilebase = opts.outfilebase
f = {}
for ant in range(len(h5.ants)):
    name = h5.ants[ant].name
    f[name] = file('%s_%s.csv'%(outfilebase,h5.ants[ant].name), 'w')
    f[name].write('# antenna = %s\n' % h5.ants[ant].description)
    f[name].write(', '.join(output_field_names) + '\n')

for cscan in h5.compscans() :
    print("Compound scan %i of field %s "%(cscan[0],cscan[2].name) )
    offset_data = reduce_compscan_inf(h5,channel_mask)
    if len(offset_data) > 0 : # if not an empty set
        print("Valid data obtained from the Compound scan")
        for antname in offset_data:
            f[antname].write(output_fields % offset_data[antname])
            f[antname].flush() # Because I like to see stuff in the file
      
for ant in range(len(h5.ants)):
    name = h5.ants[ant].name
    f[name].close()
