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

def active_ants(ds, state, pct=10, rel_pct=80):
    """ Find antennas  for which at least `pct` of dumps have activity=`state` and the worst has no less
        than `rel_pct` the counts of the best.
        @param ds: a katdal dataset.
        @return: list of antenna names that can be considered as being in activity=`state`"""
    ants = []
    activity = []
    for ant in ds.ants:
        ants.append(ant.name)
        activity.append(np.count_nonzero(ds.sensor["%s_activity"%ant.name] == state))
    fracs = np.asarray(activity)/float(ds.shape[0])
    rel_fracs = np.asarray(activity)/np.max(activity)
    ants = np.asarray(ants)[(fracs >= pct/100.) & (rel_fracs >= rel_pct/100.)]
    return ants

#TODO Remove this function once katdal has this functionality 
def activity(h5,state = 'track'):
    """Strict Array Activity Sensor: because some of antennas have a mind of their own, 
    others appear to have lost theirs entirely.
        @return: boolean flags per dump, True when ALL antennas match `state` and nd_coupler=False """
    antlist = [a.name for a in h5.ants]
    activityV = np.zeros((len(antlist),h5.shape[0]) ,dtype=bool)
    for i,ant in enumerate(antlist) :
        sensor = h5.sensor['%s_activity'%(ant)] ==state
        if ~np.any(sensor):
            print("Antenna %s has no valid %s data"%(ant,state))
        noise_diode = ~h5.sensor['Antennas/%s/nd_coupler'%(ant)]
        activityV[i,:] +=   noise_diode &  sensor
    return np.all(activityV,axis=0)

def w_average(arr,axis=None, weights=None):
    return np.nansum(arr*weights,axis=axis)/np.nansum(weights,axis=axis)

class inputvalue:
    pass

def fit_beam(pos,gains_p,stdv,rfi_static_flags,target, temperature, pressure, humidity,middle_time,chunks=16,valid_offset=np.pi/8):
    for pol in gains_p.keys():
        pass
    gaussian_centre     = np.full((chunks * 2, 2, gains_p[pol].shape[-1]), np.nan)
    gaussian_centre_std = np.full((chunks * 2, 2, gains_p[pol].shape[-1]), np.nan)
    gaussian_width      = np.full((chunks * 2, 2, gains_p[pol].shape[-1]), np.nan)
    gaussian_width_std  = np.full((chunks * 2, 2, gains_p[pol].shape[-1]), np.nan)
    gaussian_height     = np.full((chunks * 2, gains_p[pol].shape[-1]), np.nan)
    gaussian_height_std = np.full((chunks * 2, gains_p[pol].shape[-1]), np.nan)
    freq = np.zeros((1024),bool)
    for i,pol in enumerate(gains_p.keys()) :
        for ant in range(gains_p[pol].shape[-1]):
            for chunk in range(chunks):
                freq[:] = False
                freq[chunk*(h5.shape[1]//chunks):(chunk+1)*(h5.shape[1]//chunks)] = True
                rfi = ~rfi_static_flags & freq
                if (pos[pol].shape[0] > 4) and np.any(rfi): # Make sure there is enough data for a fit
                    fitobj  = fit.GaussianFit(pos[pol][:,:,ant].mean(axis=0),[1.,1.],1)
                    x = np.column_stack((pos[pol][:,0,ant],pos[pol][:,1,ant]))
                    y = np.abs(gains_p[pol][:,rfi,ant]).mean(axis=1)
                    y_err = 1./np.sqrt(stdv[pol][:,rfi,ant].sum(axis=1))
                    gaussian = fitobj.fit(x.T,y,y_err )
                    # Fitted beam center is in (x, y) coordinates, in projection
                    # centred on beam using Swaped Sin projection , have to do
                    # this a second time to cancel out the swap
                    snr = np.abs(np.r_[gaussian.std/gaussian.std_std])
                    valid_fit = np.all(np.isfinite(np.r_[gaussian.mean,gaussian.std_mean,gaussian.std,gaussian.std_std,gaussian.height,gaussian.std_height,snr]))
                    theta =  np.sqrt((gaussian.mean**2).sum())  # Cart. distance calc
                    # This is to see if the co-ord is out of range
                    # The valid fit is needed because I have no way of working out if the gain solution was ok.
                    if valid_fit and np.any(theta <= valid_offset) : # Invalid fits remain nan (initialised defaults)
                        # Convert this offset back to spherical (az, el) coordinates
                        beam_center_azel = target.plane_to_sphere(np.radians(gaussian.mean[0]), np.radians(gaussian.mean[1]), middle_time, antenna=h5.ants[ant],projection_type="SSN")
                        # Now correct the measured (az, el) for refraction and then apply the old pointing model
                        # to get a "raw" measured (az, el) at the output of the pointing model
                        rc = katpoint.RefractionCorrection()
                        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
                        beam_center_azel = h5.ants[ant].pointing_model.apply(*beam_center_azel)
                        beam_center_azel = np.degrees(np.array(beam_center_azel))
                        gaussian_centre[chunk+i*chunks,:,ant]     = beam_center_azel
                        gaussian_centre_std[chunk+i*chunks,:,ant] = gaussian.std_mean
                        gaussian_width[chunk+i*chunks,:,ant]      = gaussian.std
                        gaussian_width_std[chunk+i*chunks,:,ant]  = gaussian.std_std
                        gaussian_height[chunk+i*chunks,ant]       = gaussian.height
                        gaussian_height_std[chunk+i*chunks,ant]   = gaussian.std_height
    returnval = inputvalue()
    returnval.centre = gaussian_centre
    returnval.centre_std = gaussian_centre_std
    returnval.width = gaussian_width
    returnval.width_std = gaussian_width_std
    returnval.height = gaussian_height
    returnval.height_std = gaussian_height_std
    return returnval

def reduce_compscan_inf(h5,rfi_static_flags=None,chunks=16,use_weights=False,compscan_index=None):
    """Break the band up into chunks"""
    chunk_size = chunks
    rfi_static_flags = np.full(h5.shape[1], False) if (rfi_static_flags is None) else rfi_static_flags
    gains_p = {}
    pos = {}
    stdv = {}
    calibrated = False # placeholder for calibration
    katpoint.projection.set_out_of_range_treatment('nan')
    h5.target_projection='SSN'

    h5.select(compscans=compscan_index)
    # Combine target indices if they refer to the same target for the purpose of this analysis
    TGT = h5.catalogue.targets[h5.target_indices[0]].description.split(",")
    def _eq_TGT_(tgt): # tgt==TGT, "tags" don't matter
        tgt = tgt.description.split(",")
        return (tgt[0] == TGT[0]) and (tgt[2] == TGT[2]) and (tgt[3] == TGT[3])

    target_indices = [TI for TI in h5.target_indices if _eq_TGT_(h5.catalogue.targets[TI])]
    target_list = [h5.catalogue.targets[TI] for TI in target_indices]
    if len(h5.target_indices) > len(target_indices):
        print("Warning multiple targets in the compscan, using %s instead of %s"%(target_indices,h5.target_indices))
    target = h5.catalogue.targets[h5.target_indices[0]]
    
    # Calculate average target flux over entire band
    flux_spectrum = h5.catalogue.targets[h5.target_indices[0]].flux_density(h5.freqs) # include flags
    average_flux = np.mean([flux for flux in flux_spectrum if not np.isnan(flux)])
    temperature = np.mean(h5.temperature)
    pressure = np.mean(h5.pressure)
    humidity = np.mean(h5.humidity)
    wind_speed = np.mean(h5.wind_speed)
    wind_direction  = np.degrees(np.angle(np.mean(np.exp(1j*np.radians(h5.wind_direction)))) )# Vector Mean
    sun = katpoint.Target('Sun, special')
    # Obtain middle timestamp of compound scan, where all pointing calculations are done
    middle_time = np.median(h5.timestamps[:], axis=None)
    # Calculate pointing offset
    # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
    #requested_azel = target.azel(middle_time)
    # Correct for refraction, which becomes the requested value at input of pointing model
    #rc = katpoint.RefractionCorrection()
    #requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
    #requested_azel = katpoint.rad2deg(np.array(requested_azel))

   
    pols = ["H","V"] # Put in logic for Intensity
    for i,pol in enumerate(pols) :
        gains_p[pol] = []
        pos[pol] = []
        stdv[pol] = []
        h5.select(pol=pol,corrprods='cross',ants=h5.antlist,targets=target_list,compscans=compscan_index)
        h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
        for scan in h5.scans() :
            if scan[1] != 'track':               continue
            valid_index = activity(h5,state = 'track')
            data = h5.vis[valid_index]
            if data.shape[0] > 0 : # need at least one data point
                if use_weights :
                    weights = h5.weights[valid_index].mean(axis=0)
                else:
                    weights = np.ones(data.shape[1:]).astype(float)

                gains_p[pol].append(calprocs.g_fit(data[:].mean(axis=0),weights,h5.bls_lookup,refant=0) )
                stdv[pol].append(np.ones((data.shape[0],data.shape[1],len(h5.ants))).sum(axis=0))#number of data points
                # Get coords in (x(time,ants),y(time,ants) coords) 
                pos[pol].append( [h5.target_x[valid_index,:].mean(axis=0), h5.target_y[valid_index,:].mean(axis=0)] )
        gains_p[pol] = np.array(gains_p[pol])
        stdv[pol] = np.array(stdv[pol])
        pos[pol] = np.array(pos[pol]) # Maybe one day there could be a correction for the squint

    gaussian = fit_beam(pos,gains_p,stdv,rfi_static_flags,target, temperature, pressure, humidity,middle_time) # fit beam

    ant_pointing = {}
    pols = ["HH","VV",'I']
    pol_ind = {}
    pol_ind['HH'] = np.arange(0.0*chunk_size,1.0*chunk_size,dtype=int) # H is the first half
    pol_ind['VV'] = np.arange(1.0*chunk_size,2.0*chunk_size,dtype=int) # V is the second half
    pol_ind['I']  = np.arange(0.0*chunk_size,2.0*chunk_size,dtype=int) # I is all the samples.
    for ant in range(len(h5.ants)):
        valid_solutions = 0
        if gaussian.centre.shape[2] > 0 : # If there are no gains then no 3rd axis returend from beam fit
            I_sol = ~np.isnan(gaussian.centre[pol_ind['I'],:,ant]) & ~np.isnan(gaussian.centre_std[pol_ind['I'],:,ant])
            valid_solutions = np.count_nonzero(I_sol)//2 # Note this is four the number of solutions because of the Az & El parts and the H & V
        print("%i valid solutions out of %s for %s on %s at %s "%(valid_solutions//2,chunks,h5.ants[ant].name,target.name,str(katpoint.Timestamp(middle_time))))
        if valid_solutions//2 > 0 : # a bit overboard
            name = h5.ants[ant].name
            ant_pointing[name] = {}
            ant_pointing[name]["antenna"] = h5.ants[ant].name
            ant_pointing[name]["valid_solutions"] = valid_solutions
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
            azel_beam = w_average(gaussian.centre[pol_ind["I"],:,ant],axis=0,weights=1./gaussian.centre_std[pol_ind["I"],:,ant]**2)
            # Make sure the offset is a small angle around 0 degrees
            offset_azel = katpoint.wrap_angle(azel_beam - requested_azel, 360.)
            ant_pointing[name]["delta_azimuth"] =offset_azel.tolist()[0]
            ant_pointing[name]["delta_elevation"] =offset_azel.tolist()[1]
            ant_pointing[name]["delta_elevation_std"] =0.0#calc
            ant_pointing[name]["delta_azimuth_std"] =0.0#calc
            for pol in pol_ind:
                ant_pointing[name]["beam_height_%s"%(pol)]     = w_average(gaussian.height[pol_ind[pol],ant],axis=0,weights=1./gaussian.height_std[pol_ind[pol],ant]**2)
                ant_pointing[name]["beam_height_%s_std"%(pol)] = np.sqrt(np.nansum(gaussian.height_std[pol_ind[pol],ant]**2) )
                ant_pointing[name]["beam_width_%s"%(pol)]      = w_average(gaussian.width[pol_ind[pol],:,ant],axis=0,weights=1./gaussian.width_std[pol_ind[pol],:,ant]**2).mean() 
                ant_pointing[name]["beam_width_%s_std"%(pol)]  = np.sqrt(np.nansum(gaussian.width_std[pol_ind[pol],:,ant]**2) )
                ant_pointing[name]["baseline_height_%s"%(pol)] = 0.0
                ant_pointing[name]["baseline_height_%s_std"%(pol)] = 0.0
                ant_pointing[name]["refined_%s"%(pol)] =  5.0  # I don't know what this means
                ant_pointing[name]["azimuth_%s"%(pol)]       =w_average(gaussian.centre[pol_ind[pol],0,ant],axis=0,weights=1./gaussian.centre_std[pol_ind[pol],0,ant]**2)
                ant_pointing[name]["elevation_%s"%(pol)]     =w_average(gaussian.centre[pol_ind[pol],1,ant],axis=0,weights=1./gaussian.centre_std[pol_ind[pol],1,ant]**2)
                ant_pointing[name]["azimuth_%s_std"%(pol)]   =np.sqrt(np.nansum(gaussian.centre_std[pol_ind[pol],0,ant]**2) )
                ant_pointing[name]["elevation_%s_std"%(pol)] =np.sqrt(np.nansum(gaussian.centre_std[pol_ind[pol],1,ant]**2) )
    return ant_pointing



def load_rfi_static_mask(filename, freqs, debug_chunks=0):
    # Construct a mask either from a pickle file, or a text file with frequency ranges
    nchans = len(freqs)
    channel_width = abs(freqs[1]-freqs[0])
    try:
        with open(filename, "rb") as pickle_file:
            channel_flags = pickle.load(pickle_file)
        nflags = len(channel_flags)
        if (nchans != nflags):
            print("Warning channel mask (%d) is stretched to fit dataset (%d)!"%(nflags,nchans))
            N = nchans/float(nflags)
            channel_flags = np.repeat(channel_flags, int(N+0.5)) if (N > 1) else channel_flags[::int(1/N)]
        channel_flags = channel_flags[:nchans] # Clip, just in case
    except pickle.UnpicklingError: # Not a pickle file, perhaps a plain text file with frequency ranges in MHz?
        mask_ranges = np.loadtxt(filename, comments='#', delimiter=',')
        channel_flags = np.full((nchans,), False)
        low = freqs - 0.5 * channel_width
        high = freqs + 0.5 * channel_width
        for r in mask_ranges:
            in_range = (low <= r[1]*1e6) & (r[0]*1e6 <= high)
            idx = np.where(in_range)[0]
            channel_flags[idx] = True
    if debug_chunks > 0:
        for chunk in range(debug_chunks):
            freq = slice(chunk*(nchans//debug_chunks),(chunk+1)*(nchans//debug_chunks))
            masked_f = freqs[freq][channel_flags[freq]]
            if (len(masked_f) > 0):
                print("\tFreq. chunk %d: mask omits (%.1f - %.1f)MHz"%(chunk,np.min(masked_f)/1e6,np.max(masked_f)/1e6))
            else:
                print("\tFreq. chunk %d: mask omits nothing"%chunk)
    return channel_flags


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
parser.add_option( "--use-weights",action="store_true",
                  default=False, help="Use SDP visability weights ")

parser.add_option("-c", "--channel-mask", default="/var/kat/katsdpscripts/RTS/rfi_mask.pickle", help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
parser.add_option("-o", "--output", dest="outfilebase",default=None,
                  help="Base name of output files (*.csv for output data and *.log for messages, "
                       "default is '<dataset_name>_interferometric_pointing')")

(opts, args) = parser.parse_args()

chunks = 16 # Default

output_fields = '%(dataset)s, %(target)s, %(timestamp_ut)s, %(azimuth).7f, %(elevation).7f, ' \
                '%(delta_azimuth).7f, %(delta_azimuth_std).7f, %(delta_elevation).7f, %(delta_elevation_std).7f, ' \
                '%(data_unit)s, %(beam_height_I).7f, %(beam_height_I_std).7f, %(beam_width_I).7f, ' \
                '%(beam_width_I_std).7f, %(baseline_height_I).7f, %(baseline_height_I_std).7f, %(refined_I).7f, ' \
                '%(beam_height_HH).7f, %(beam_width_HH).7f, %(baseline_height_HH).7f, %(refined_HH).7f, ' \
                '%(beam_height_VV).7f, %(beam_width_VV).7f, %(baseline_height_VV).7f, %(refined_VV).7f, ' \
                '%(frequency).7f, %(flux).4f, %(temperature).2f, %(pressure).2f, %(humidity).2f, %(wind_speed).2f, ' \
                '%(wind_direction).2f , %(sun_az).7f, %(sun_el).7f, %(timestamp)i, %(valid_solutions)i \n'

output_field_names = [name.partition(')')[0] for name in output_fields[2:].split(', %(')]

h5 = katdal.open(args[0])  
ant_list = list(active_ants(h5, 'track')) # Default list only includes those that 'track'ed some of the time
if opts.ants is not None  :
    ant_list = opts.ants.split(',')
if opts.ex_ants is not None :
    for ant in opts.ex_ants.split(','):
        if ant in ant_list:
            ant_list.remove(ant)
h5 = katdal.open(args[0])
print("Using '%s' as the reference antenna "%(h5.ref_ant))
h5.select(compscans='interferometric_pointing',ants=ant_list)

if len(opts.channel_mask)>0:
    rfi_static_flags = load_rfi_static_mask(opts.channel_mask, h5.freqs, debug_chunks=chunks)
else:
    rfi_static_flags = None

h5.antlist = [a.name for a in h5.ants]
h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
if opts.outfilebase is None :
    outfilebase =  "%s_%s"%(h5.name.split('/')[-1].split('.')[0], "interferometric_pointing")
else:
    outfilebase = opts.outfilebase
f = {}
for ant in range(len(h5.ants)):
    name = h5.ants[ant].name
    f[name] = open('%s_%s.csv'%(outfilebase,h5.ants[ant].name), 'w')
    f[name].write('# antenna = %s\n' % h5.ants[ant].description)
    f[name].write(', '.join(output_field_names) + '\n')
for compscan_index  in h5.compscan_indices :
    print("Compound scan %i  "%(compscan_index) )
    offset_data = reduce_compscan_inf(h5,rfi_static_flags,chunks,use_weights=opts.use_weights,compscan_index=compscan_index)
    if len(offset_data) > 0 : # if not an empty set
        print("Valid data obtained from the Compound scan")
        for antname in offset_data:
            f[antname].write(output_fields % offset_data[antname])
            f[antname].flush() # Because I like to see stuff in the file
    
for ant in range(len(h5.ants)):
    name = h5.ants[ant].name
    f[name].close()
