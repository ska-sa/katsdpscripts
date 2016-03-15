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

def get_condition(data):
    """Get condition for grouped target scan.
    'ideal'   = 0 , \n 'optimal' = 1, \n 'normal'  = 2, \n 'other'   = 3 """
    # Set up limits on environmental conditions
    condition_values = np.zeros((3), dtype=dict)
    condition_values[0] = {'wind_speed':1.,'temp_low':19.,'temp_high':21.,'sun_el':-5.} #  ideal
    condition_values[1] = {'wind_speed':2.9,'temp_low':-5.,'temp_high':35.,'sun_el':-5.}#  optimal
    condition_values[2] = {'wind_speed':9.8,'temp_low':-5.,'temp_high':40.,'sun_el':100.}# normal
    condition_values[3] = {'wind_speed':9999.8,'temp_low':-273.,'temp_high':40000.,'sun_el':1000.}# other
    for i,values in enumerate(condition_values)
        condition = i
        if data['sun_el'].max() < condition_values[i]['sun_el'] :
            if data['wind_speed'].max() < condition_values[i]['wind_speed'] :
                if data['temperature'].max() < condition_values[i]['temp_high'] :
                    if data['temperature'].min() > condition_values[i]['temp_low'] :
                        break # Means conditions have been met 
    return condition

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

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


def referencemetrics(ant,data,num_samples_limit=1):
    """Determine and sky RMS from the antenna pointing model."""
    """On the calculation of all-sky RMS
     Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
     They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
     standard deviation of sigma
     The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
     The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
     two squared Gaussian random values, each with an expected value of sigma^2.
      e.g. sky_rms = np.sqrt(np.ma.mean((abs_sky_error-abs_sky_error.mean()) ** 2))

     A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
     which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
      e.g. robust_sky_rms = np.ma.median(np.sqrt((abs_sky_error-abs_sky_error.mean())**2)) * np.sqrt(2. / np.log(4.))
    """
    text = [] #azimuth, elevation, delta_azimuth, delta_azimuth_std, delta_elevation, delta_elevation_std,
    measured_delta_xel  =  data['delta_azimuth']* np.cos(data['elevation']) # scale due to sky shape
    abs_sky_error = np.ma.array(data=measured_delta_xel,mask=False)
    model_delta_az, model_delta_el = ant.pointing_model.offset(data['azimuth'], data['elevation'])
    residual_az = data['delta_azimuth']   - model_delta_az
    residual_el = data['delta_elevation'] - model_delta_el
    residual_xel = residual_az * np.cos(data['elevation'])
    delta_xel_std = data['delta_azimuth_std'] * np.cos(data['elevation'l)
    abs_sky_delta_std = rad2deg(np.sqrt(delta_xel_std**2 + data['delta_azimuth_std']**2))*3600 # make arc seconds
    
    #print ("Test Target: '%s'   fit accuracy %.3f\"  "%(target,abs_sky_delta_std[key])) 
        
    abs_sky_error = rad2deg(np.sqrt((residual_xel) ** 2 + (residual_el)** 2)) *3600
    if data.shape[0]  > num_samples_limit: # check all fitted Ipks are valid
        #TODO get_condition(data)

    if data.shape[0]  > num_samples_limit:
        rms = np.std(abs_sky_error)
        #TODO get_condition(data)
        text.append("Dataset:%s  Test Target: '%s' Reference RMS = %.3f\" {fit-accuracy=%.3f\"} (robust %.3f\")  (N=%i Data Points) ['%s']" % (data['dataset'][0],
            target,np.std(abs_sky_error),np.mean(abs_sky_delta_std),np.ma.median(np.abs(abs_sky_error-abs_sky_error.mean())) * np.sqrt(2. / np.log(4.)),keep.sum(),condition))

        # get the (environmental) conditions for each grouped target scan
        fitIpks = np.append(fitIpks, condArray['beam_height_I'])
        if ( condition == 'normal' ):
            normIndices = np.append(normIndices,index)
        elif ( condition == 'optimal' ):
            optIndices = np.append(optIndices,index)
        elif ( condition == 'ideal' ):
            idealIndices = np.append(idealIndices,index)
    sky_rms = np.sqrt(np.ma.mean((abs_sky_error-abs_sky_error.mean()) ** 2))
    robust_sky_rms = np.ma.median(np.sqrt((abs_sky_error-abs_sky_error.mean())**2)) * np.sqrt(2. / np.log(4.))
    text.append("Dataset:%s  All Sky Reference RMS = %.3f\" (robust %.3f\")   (N=%i Data Points) R.T.P.4"  % (data['dataset'][0],sky_rms, robust_sky_rms,abs_sky_error.count()))
    #TODO output_data
    return text,output_data

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



class group():
    """This is an class to make an itterater that go's through the array and returns data in chuncks"""
    def __init__(self.obj):
        self.data = obj
    
    def __iter__(self):
        index_list = [0,1,2,3]
        while len(index_list) > 0 :
            yield self.data[index_list]

            
# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')

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

data = None
for filename in args:
    if data is None:
        data,ant = read_offsetfile(filename)
        #offsetdata = data
    else:
        tmp_offsets,tmp_ant = read_offsetfile(filename)
        data = np.r_[data,tmp_offsets]
        if not ant == tmp_ant : raise RuntimeError('The antenna has changed')
        #offsetdata = data

# fix units and wraps
data['azimuth'],data['elevation']  = angle_wrap(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
data['delta_azimuth'], data['delta_elevation']= deg2rad(data['delta_azimuth']), deg2rad(data['delta_elevation'])
data['delta_azimuth_std'], deg2rad(data['delta_elevation_std'] = deg2rad(data['delta_azimuth_std']), deg2rad(data['delta_elevation_std'])

for offsetdata in group(data) : 
    #New loop to provide the data in steps of test offet scans .
    text,output = referencemetrics(ant,offsetdata,opts.num_samples_limit)
    textString += text
    textString.append("") # new line when joined 
    #if ( dangText != [] ):
    #    dText += dangText

# create source and condition-separated RMS arrays
for targ in uniqTargets:
    pass
        #norm_sourceSepTOD[targ] = np.float32(allMJDs[normIndices])%1*24

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