import sys, optparse, logging, time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,AutoMinorLocator
from astropy.time import Time
import katpoint
from katpoint import rad2deg, deg2rad, Target
from katsdpscripts import git_info
from katsdpscripts.RTS.weatherlib import select_and_average, select_environment, rolling_window
from matplotlib.offsetbox import AnchoredText

def get_condition(data,ant,source):
    """Get condition for grouped target scan."""
    # Set up limits on environmental conditions
    ideal = {'wind_speed':1.,'temp_low':19.,'temp_high':21.,'sun_el':-5.}
    optimal = {'wind_speed':2.9,'temp_low':-5.,'temp_high':35.,'sun_el':-5.}
    normal = {'wind_speed':9.8,'temp_low':-5.,'temp_high':40.,'sun_el':100.}

    condArray = np.zeros(7,dtype=np.float32)
    flagArray = np.ones(3,dtype=bool)
    boolArray = np.zeros(data['target'].size,dtype=bool)
    indices = np.where(data['target']==source)[0]
    condArray[0] = data['elevation'][indices].mean()
    fitIpks = data['beam_height_I'][indices]
    medIpk = np.median(fitIpks)
    acceptIndices = np.where((fitIpks >= medIpk*0.8)&(fitIpks<=medIpk*1.2))[0]
    boolArray[indices[acceptIndices]] = 1
    condArray[0] = data['elevation'][indices].mean()
    condArray[6] = data['beam_height_I'][indices].mean()
    keys = ['elevation','timestamp','wind_speed','sun_el','temperature','sun_angle','beam_height_I']
    for key in keys[1:5]:
        if ( key == 'timestamp' ):
            try:
                avTstamp = data[key][indices].mean()
            except(TypeError,ValueError):
                utcs = data['timestamp_ut'][indices]
                mjds = Time(utcs,format='iso',scale='utc').mjd 
                timestamps = np.float128(mjds - 40587.0)*86400.0
                avTstamp = timestamps.mean() 
            condArray[1] = avTstamp
        else:
            if ( key == 'wind_speed' ):
                windLims = np.array([normal[key],optimal[key],ideal[key]])
                flagArray = flagArray & (data[key][indices].mean() < windLims)
                condArray[2] = data[key][indices].mean()

            if ( key == 'sun_el' ):
                sunLims = np.array([normal[key],optimal[key],ideal[key]])
                source = katpoint.construct_azel_target(np.radians(data['azimuth'][indices].mean()),
                                                    np.radians(data['elevation'][indices].mean()))
                try:
                    sun_az = np.radians(data['sun_az'][indices].mean())
                    sun_el = np.radians(data['sun_el'][indices].mean())
                    sun = katpoint.construct_azel_target(sun_az,sun_el,antenna=ant)  
                except(TypeError,ValueError):
                    sun = Target('Sun,special')
                    sun_el = np.degrees(sun.azel(timestamp=avTstamp,antenna=ant))[1]
                flagArray = flagArray & (sun_el < sunLims)
                condArray[3] = sun_el
                condArray[5] = np.degrees(source.separation(sun,timestamp=avTstamp,antenna=ant))
            
            if ( key == 'temperature' ):
                lowLims = np.array([normal['temp_low'],optimal['temp_low'],ideal['temp_low']])
                highLims = np.array([normal['temp_high'],optimal['temp_high'],ideal['temp_high']])
                condArray[4] = data[key][indices].mean()
                flagArray = flagArray & (data[key][indices].mean() > lowLims) & (data[key][indices].mean() < highLims)

    # environmental condition test
    try:
        condIndex = np.where(flagArray==True)[0][-1]
        condition = ['normal','optimal','ideal'][condIndex]
    except(IndexError):
        condition = 'bad'
    return boolArray, condition, np.rec.fromarrays(condArray,dtype=zip(keys,np.tile(np.float,7)))

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

def save_pointingmodel(filebase,model):
    # Save pointing model to file
    outfile = file(filebase + '.csv', 'w')
    outfile.write(model.description)
    outfile.close()
    logger.debug("Saved %d-parameter pointing model to '%s'" % (len(model.params), filebase + '.csv'))

# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')

def check_target(target,calFile):
    """Check target name and modify if necessary to remove duplicates."""
    calibrators = np.loadtxt(calFile,usecols=[0],delimiter=',',dtype=str)
    sources = np.array([str(cal).replace('*','').split('|')[0].strip(' ') for cal in calibrators])
    names = np.array([str(cal).replace('*','').split('|')[-1].strip(' ') for cal in calibrators])
    index = np.where(sources==target)[0]
    if ( len(index) == 1):
        source = names[index][0]
    else:
        source = target
    return source

def read_offsetfile(filename):
    # Load data file in one shot as an array of strings
    string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
    data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
    # Interpret first non-comment line as header
    fields = data[0].tolist()
    # By default, all fields are assumed to contain floats
    formats = np.tile(np.float, len(fields))
    # The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
    formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
    # Convert to heterogeneous record array
    data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))
    # Load antenna description string from first line of file and construct antenna object from it
    antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
    # Use the pointing model contained in antenna object as the old model (if not overridden by file)
    # If the antenna has no model specified, a default null model will be used
    return data, antenna

def referencemetrics(index,ant,az, el,measured_delta_az, measured_delta_el,delta_azimuth_std=0,delta_elevation_std=0,num_samples_limit=1,cal_file=None):
    """Determine and sky RMS from pointing model."""
    text = []
    danger_text = []
    measured_delta_xel  =  measured_delta_az* np.cos(el) # scale due to sky shape
    abs_sky_error = np.ma.array(data=measured_delta_xel,mask=False)
    model_delta_az, model_delta_el = ant.pointing_model.offset(az, el)
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel = residual_az * np.cos(el)
    
    delta_xel_std = delta_azimuth_std * np.cos(el)
    abs_sky_delta_std = rad2deg(np.sqrt(delta_xel_std**2 + delta_azimuth_std**2))*3600
    
    elevs = np.array([])
    mjds = np.array([])
    windSpeed = np.array([])
    sunAngles = np.array([])
    fitIpks = np.array([])
    temps = np.array([])
    skyRMS = np.array([])
    sources = np.array([],dtype=str)
    normIndices = np.array([],dtype=int)
    optIndices = np.array([],dtype=int)
    idealIndices = np.array([],dtype=int)
    for target in set(offsetdata['target']):  # ascertain target group condition
        keep = np.ones((len(offsetdata)),dtype=np.bool)
        for key,targetv in enumerate(offsetdata['target']):
            keep[key] = target == targetv
            if keep[key] and opts.no_plot: 
                print ("Test Target: '%s'   fit accuracy %.3f\"  "%(target,abs_sky_delta_std[key])) 
        
        abs_sky_error[keep] = rad2deg(np.sqrt((residual_xel[keep]) ** 2 + (residual_el[keep])** 2)) *3600
        if keep.sum() > num_samples_limit: # check all fitted Ipks are valid
            boolArray,condition,condArray = get_condition(offsetdata,ant,target)
            keep = np.copy(boolArray)

        if keep.sum() > num_samples_limit:
            rms = np.std(abs_sky_error[keep])
            useIndices,condition,condArray = get_condition(offsetdata,ant,target)
            source = check_target(target,cal_file)
            text.append("Dataset:%s  Test Target: '%s' Reference RMS = %.3f\" {fit-accuracy=%.3f\"} (robust %.3f\")  (N=%i Data Points) ['%s']" % (offsetdata['dataset'][0],
                target,np.std(abs_sky_error[keep]),np.mean(abs_sky_delta_std[keep]),np.ma.median(np.abs(abs_sky_error[keep]-abs_sky_error[keep].mean())) * np.sqrt(2. / np.log(4.)),keep.sum(),condition))
    
            if ( rms > 80 ):
                danger_text.append("Dataset:%s  Test Target: '%s' Reference RMS = %.3f\" {fit-accuracy=%.3f\"} (robust %.3f\")  (N=%i Data Points) ['%s']" % (offsetdata['dataset'][0],
                    target,np.std(abs_sky_error[keep]),np.mean(abs_sky_delta_std[keep]),np.ma.median(np.abs(abs_sky_error[keep]-abs_sky_error[keep].mean())) * np.sqrt(2. / np.log(4.)),keep.sum(),condition))

            # get the (environmental) conditions for each grouped target scan
            elevs = np.append(elevs,condArray['elevation'])
            mjds = np.append(mjds,np.float128(np.float128(condArray['timestamp']/np.float128(86400.0)+40587.0)))
            windSpeed = np.append(windSpeed,condArray['wind_speed'])
            sunAngles = np.append(sunAngles,condArray['sun_angle'])
            fitIpks = np.append(fitIpks, condArray['beam_height_I'])
            temps = np.append(temps,condArray['temperature']) 
            skyRMS = np.append(skyRMS,rms)
            sources = np.append(sources,source)
            if ( condition == 'normal' ):
                normIndices = np.append(normIndices,index)
            elif ( condition == 'optimal' ):
                optIndices = np.append(optIndices,index)
            elif ( condition == 'ideal' ):
                idealIndices = np.append(idealIndices,index)
            index += 1
        else : 
            abs_sky_error.mask[keep] = True # Remove Samples from the catalogue if there are not enough mesuments
    ###### On the calculation of all-sky RMS #####
    # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
    # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
    # standard deviation of sigma
    # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
    # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
    # two squared Gaussian random values, each with an expected value of sigma^2.
    sky_rms = np.sqrt(np.ma.mean((abs_sky_error-abs_sky_error.mean()) ** 2))
    #print abs_sky_error
    # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
    # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
    robust_sky_rms = np.ma.median(np.sqrt((abs_sky_error-abs_sky_error.mean())**2)) * np.sqrt(2. / np.log(4.))
    text.append("Dataset:%s  All Sky Reference RMS = %.3f\" (robust %.3f\")   (N=%i Data Points) R.T.P.4"  % (offsetdata['dataset'][0],sky_rms, robust_sky_rms,abs_sky_error.count()))
    return text,index,[elevs,mjds,windSpeed,sunAngles,temps,fitIpks],normIndices,optIndices,idealIndices,skyRMS,sources,danger_text

def plot_source_rms(tods,fitIpks,sunAngles,skyRMS,title):
    """Plot source pointing accuracy vs sun angles."""
    fig = plt.figure(figsize=(16,9))
    params = {'axes.labelsize': 12, 'font.size': 10, 'legend.fontsize': 9, 
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': False}
    plt.rcParams.update(params)

    index = 0
    ymax = 0
    cVals = np.array(['k','b','r','g','m']*2)
    colours = np.r_[cVals,np.roll(cVals,-1),np.roll(cVals,-2)] # good up to 30 sources
    mecVals = np.copy(colours)
    fcVals = np.copy(colours)
    fcVals[np.array([11,12,13,14,16,17,18,19])] = 'w'
    markers = ['x','o','s','*','^','+','D','v','<','>']*3
    targets = sunAngles.keys()

    # sky RMS vs sun angles
    ax = fig.add_subplot(121)
    for targ in targets:
        if ( skyRMS[targ] != None ):
            plt.plot(sunAngles[targ],skyRMS[targ],mec=mecVals[index],mfc=fcVals[index],
                            marker=markers[index],color=colours[index],lw=0,label=targ)
            if ( skyRMS[targ].max() > ymax ):
                ymax = skyRMS[targ].max()
                ymax = ymax - ymax%5 + 10
            index += 1

    plt.suptitle(title, fontsize=12, fontweight='bold',y=0.95)
    plt.legend(numpoints=1,loc='upper right')
    plt.axhline(y=5,color='g',lw=1,ls='--')
    plt.axhline(y=25,color='b',lw=1,ls='--')
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Sun Angular Distance (deg)')
    if ( ymax > 80 ):
        plt.ylim(0,80)  # arbitrary y-axis limit for visualisation purposes
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # fitted Gaussian peak height vs MJD
    index = 0
    ax2 = fig.add_subplot(122)
    for targ in targets:
        if ( fitIpks[targ] != None ):
            plt.plot(tods[targ],fitIpks[targ],mec=mecVals[index],mfc=fcVals[index],
                            marker=markers[index],color=colours[index],lw=0,label=targ)
            index += 1
    #plt.legend(numpoints=1,loc='upper right')
    plt.ylabel(r'Fitted $I_{\mathrm{peak}}$ (A.U.)')
    plt.xlabel('Time of Day (hr)')
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    return fig

def plot_diagnostics(condArray,flagArray,all_skyRMS,title):
    """Plot pointing accuracy vs environmental conditions."""
    fig = plt.figure(figsize=(16,9))
    params = {'axes.labelsize': 12, 'font.size': 10, 'legend.fontsize': 9, 
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': False}
    plt.rcParams.update(params)
    
    colours = ['b', 'g', 'y']
    markers = ['o','s','^']
    labels = ['normal','optimal','ideal']
    sizes = np.array([np.size(indices) for indices in flagArray])
    indices = np.where(sizes!=0)[0]
    
    plt.suptitle(title, fontsize=12, fontweight='bold',y=0.95)
    ax = fig.add_subplot(231)
    for index in indices:
        plt.plot(condArray[0][flagArray[index]],all_skyRMS[flagArray[index]],marker=markers[index],
            color=colours[index],lw=0,label=labels[index])
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Elevation (deg)')
    plt.legend(numpoints=1,loc='upper right')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ( all_skyRMS.max() > 80 ):
        plt.ylim(0,80)

    ax2 = fig.add_subplot(232)
    tod = condArray[1]%1*24
    for index in indices:
        plt.plot(tod[flagArray[index]],all_skyRMS[flagArray[index]],marker=markers[index],
            color=colours[index],lw=0,label=labels[index])
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Hour (UTC)')
    plt.legend(numpoints=1,loc='upper right')
    plt.xlim(0,24)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    if ( all_skyRMS.max() > 80 ):
        plt.ylim(0,80)

    ax3 = fig.add_subplot(233)
    for index in indices:
        plt.plot(condArray[2][flagArray[index]],all_skyRMS[flagArray[index]],marker=markers[index],
            color=colours[index],lw=0,label=labels[index])
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Wind speed (m/s)')
    plt.legend(numpoints=1,loc='upper right')
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    if ( all_skyRMS.max() > 80 ):
        plt.ylim(0,80)

    ax4 = fig.add_subplot(234)
    for index in indices:
        plt.plot(condArray[3][flagArray[index]],all_skyRMS[flagArray[index]],marker=markers[index],
            color=colours[index],lw=0,label=labels[index])
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Sun Angular Distance (deg)')   
    plt.legend(numpoints=1,loc='upper right')
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    if ( all_skyRMS.max() > 80 ):
        plt.ylim(0,80)

    ax5 = fig.add_subplot(235)
    for index in indices:
        plt.plot(condArray[4][flagArray[index]],all_skyRMS[flagArray[index]],marker=markers[index],
            color=colours[index],lw=0,label=labels[index])
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel(r'Temperature ($^o$C)')  
    plt.legend(numpoints=1,loc='upper right')
    ax5.yaxis.set_minor_locator(AutoMinorLocator())
    ax5.xaxis.set_minor_locator(AutoMinorLocator())
    if ( all_skyRMS.max() > 80 ):
        plt.ylim(0,80)

    ax6 = fig.add_subplot(236)
    for index in indices:
        plt.hist(all_skyRMS[flagArray[index]],bins=np.arange(0,85,5),histtype='bar',ec='w',alpha=0.5,
            align='mid',color=colours[index],label=labels[index])
    plt.legend(numpoints=1,loc='upper right')
    plt.ylabel('Number')
    plt.xlabel(r'$\sigma$ (arc sec)')
    return fig

def write_text(textString):
    """Write out pointing accuracy text."""
    fig = plt.figure(None,figsize = (10,16))
    plt.rc('font',size=9)
    ax = fig.add_subplot(111)
    anchored_text = AnchoredText(textString, loc=2, frameon=False)
    ax.add_artist(anchored_text)
    ax.set_axis_off()
    plt.subplots_adjust(top=0.99,bottom=0,right=0.975,left=0.01)
    return ax,fig

parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This fits a pointing model to the given data CSV file"
                               " with the targets that are included in the the offset pointing csv file "
                               " "
                               " ")
parser.add_option('-c', '--cal-file', default=None,
                  help="Calibrator catalogue file to use (default = %default).")
parser.add_option('-o', '--output', dest='outfilebase', default='pointing_model_%s' % (now,),
                  help="Base name of output files (*.csv for new pointing model and *_data.csv for residuals, "
                  "default is 'pointing_model_<time>')")
parser.add_option('--num-samples-limit', default=3,
                  help="The number of valid offset measurements needed, in order to have a valid sample." )

parser.add_option('--no-plot', default=False,action='store_true',help="Produce a pdf output")
# Minimum pointing uncertainty is arbitrarily set to 1e-12 degrees, which corresponds to a maximum error
# of about 10 nano-arcseconds, as the least-squares solver does not like zero uncertainty
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
(opts, args) = parser.parse_args()

textString = [""]
dText = [r'$\bf{Exceptionally\; poor\; data\; points:}$']
if len(args) < 1 or not args[0].endswith('.csv'):
    raise RuntimeError('Correct File not passed to program. File should be csv file')

index = 0
data = None
ant = None
allElevs = np.array([])
allMJDs = np.array([])
allWindSpeed = np.array([])
allSunAngles = np.array([])
allTemps = np.array([])
allFitIpks = np.array([])
allSkyRMS = np.array([])
allTargets = np.array([],dtype=str)
allNormIndices = np.array([],dtype=int)
allOptIndices = np.array([],dtype=int)
allIdealIndices = np.array([],dtype=int)
for filename in args:
    if data is None:
        data,ant = read_offsetfile(filename)
        offsetdata = data
    else:
        tmp_offsets,tmp_ant = read_offsetfile(filename)
        data = np.r_[data,tmp_offsets]
        offsetdata= tmp_offsets
    
    az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
    measured_delta_az, measured_delta_el = deg2rad(offsetdata['delta_azimuth']), deg2rad(offsetdata['delta_elevation'])
    delta_azimuth_std,delta_elevation_std = deg2rad(offsetdata['delta_azimuth_std']), deg2rad(offsetdata['delta_elevation_std'])

    text1,index,condArray,normIndices,optIndices,idealIndices,skyRMS,sources,dangText = referencemetrics(index,ant,az,el,
        measured_delta_az, measured_delta_el,delta_azimuth_std,delta_elevation_std,opts.num_samples_limit,cal_file=opts.cal_file)
    allElevs = np.append(allElevs,condArray[0])
    allMJDs = np.append(allMJDs,condArray[1])
    allWindSpeed = np.append(allWindSpeed,condArray[2])
    allSunAngles = np.append(allSunAngles,condArray[3])
    allTemps = np.append(allTemps,condArray[4])
    allFitIpks = np.append(allFitIpks,condArray[5])    
    allTargets = np.append(allTargets,sources)
    allSkyRMS = np.append(allSkyRMS,skyRMS)

    allNormIndices = np.append(allNormIndices,normIndices)
    allOptIndices = np.append(allOptIndices,optIndices)
    allIdealIndices = np.append(allIdealIndices,idealIndices)
    textString += text1
    textString.append("")
    if ( dangText != [] ):
        dText += dangText

# create source and condition-separated RMS arrays
uniqTargets = np.unique(allTargets) # sort on targets
norm_sourceSepTOD = dict.fromkeys(uniqTargets)
norm_sourceSepIpk = dict.fromkeys(uniqTargets)
norm_sourceSepSA = dict.fromkeys(uniqTargets)
norm_sourceSepRMS = dict.fromkeys(uniqTargets) 
opt_sourceSepTOD = dict.fromkeys(uniqTargets)
opt_sourceSepIpk = dict.fromkeys(uniqTargets)
opt_sourceSepSA = dict.fromkeys(uniqTargets)
opt_sourceSepRMS = dict.fromkeys(uniqTargets)
for targ in uniqTargets:
    indices = np.where(allTargets==targ)[0]
    normIndices = np.intersect1d(indices,allNormIndices)
    optIndices = np.intersect1d(indices,allOptIndices)
    if ( normIndices.size > 0 ): 
        norm_sourceSepTOD[targ] = np.float32(allMJDs[normIndices])%1*24
        norm_sourceSepIpk[targ] = np.float32(allFitIpks[normIndices])
        norm_sourceSepSA[targ] = np.float32(allSunAngles[normIndices])
        norm_sourceSepRMS[targ] = np.float32(allSkyRMS[normIndices])
    if ( optIndices.size > 0 ):
        opt_sourceSepTOD[targ] = np.float32(allMJDs[optIndices])%1*24
        opt_sourceSepIpk[targ] = np.float32(allFitIpks[optIndices])
        opt_sourceSepSA[targ] = np.float32(allSunAngles[optIndices])
        opt_sourceSepRMS[targ] = np.float32(allSkyRMS[optIndices])

#print new_model.description
textString.append("")
textString.append(git_info() )

if not opts.no_plot :
    nice_filename =  args[0].split('/')[-1]+ '_residual_pointing_offset'
    if len(args) > 1 : nice_filename =  args[0].split('/')[-1]+'_Multiple_files' + '_residual_pointing_offset'
    pp = PdfPages(nice_filename+'.pdf')

    # plot diagnostic plots
    suptitle = '%s offset-pointing accuracy vs environmental conditions' %ant.name.upper()
    fig = plot_diagnostics([allElevs,allMJDs,allWindSpeed,allSunAngles,allTemps],
                [allNormIndices,allOptIndices,allIdealIndices],allSkyRMS,suptitle)
    fig.savefig(pp,format='pdf')
    plt.close(fig)

    # plot norm source RMS vs sun angles
    suptitle = '%s source-separated results (normal conditions)' %ant.name.upper()
    fig = plot_source_rms(norm_sourceSepTOD,norm_sourceSepIpk,norm_sourceSepSA,norm_sourceSepRMS,suptitle)
    fig.savefig(pp,format='pdf')
    plt.close(fig)

    # plot opt source RMS vs sun angles
    suptitle = '%s source-separated results (optimal conditions)' %ant.name.upper()
    fig = plot_source_rms(opt_sourceSepTOD,opt_sourceSepIpk,opt_sourceSepSA,opt_sourceSepRMS,suptitle)
    fig.savefig(pp,format='pdf')
    plt.close(fig)

    # write-out fit results
    if ( dText != [r'$\bf{Exceptionally\; poor\; data\; points:}$'] ):
        textString.append("")
        textString.append("")
        textString = np.append(np.array(textString),dText)
    ax,fig = write_text('\n'.join(textString[:110]))
    fig.savefig(pp,format='pdf')
    plt.close(fig)
    if ( len(textString) > 110 ):
        ax,fig = write_text('\n'.join(textString[110:]))
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    pp.close()
else:
    for line in textString: print line