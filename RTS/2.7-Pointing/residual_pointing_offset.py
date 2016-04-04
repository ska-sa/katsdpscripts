import optparse
import time

import numpy as np
from numpy.lib import recfunctions  # to append fields to rec arrays
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages

import katpoint
from katpoint import rad2deg, deg2rad, Target, wrap_angle
from katsdpscripts import git_info
import scipy.stats as stats

def get_condition(data):
    """Get condition for grouped target scan.
    'ideal'   = 0 , \n 'optimal' = 1, \n 'normal'  = 2, \n 'other'   = 3 """
    # Set up limits on environmental conditions
    condition_values = np.zeros((4), dtype=dict)
    condition_values[0] = {'wind_speed':1.,'temp_low':19.,'temp_high':21.,'sun_el':-5.} #  ideal
    condition_values[1] = {'wind_speed':2.9,'temp_low':-5.,'temp_high':35.,'sun_el':-5.}#  optimal
    condition_values[2] = {'wind_speed':9.8,'temp_low':-5.,'temp_high':40.,'sun_el':100.}# normal
    condition_values[3] = {'wind_speed':9999.8,'temp_low':-273.,'temp_high':40000.,'sun_el':1000.}# other
    for i,values in enumerate(condition_values) :
        condition = i
        if data['sun_el'].max() < condition_values[i]['sun_el'] :
            if data['wind_speed'].max() < condition_values[i]['wind_speed'] :
                if data['temperature'].max() < condition_values[i]['temp_high'] :
                    if data['temperature'].min() > condition_values[i]['temp_low'] :
                        break # Means conditions have been met 
    return condition


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


def referencemetrics(ant,data,num_samples_limit=1,power_sample_limit=0):
    """Determine and sky RMS from the antenna pointing model."""
    """On the calculation of all-sky RMS
     Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
     They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
     standard deviation of sigma
     The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
     The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
     two squared Gaussian random values, each with an expected value of sigma^2.
      e.g. sky_rms = np.sqrt(np.mean((abs_sky_error-abs_sky_error.mean()) ** 2))

     A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
     which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
      e.g. robust_sky_rms = np.median(np.sqrt((abs_sky_error-abs_sky_error.mean())**2)) * np.sqrt(2. / np.log(4.))
    """
    #print type(data.shape[0] ), type(num_samples_limit)
    beam = data['beam_height_I'].mean()
    good_beam = (data['beam_height_I'] > beam*.8) * (data['beam_height_I'] <  beam*1.2) * (data['beam_height_I']  > power_sample_limit)
    data = data[good_beam]
    
    if data.shape[0]  > 0 and not np.all(good_beam) : print "bad scan", data['target'][0]
    if data.shape[0]  >= num_samples_limit: # check all fitted Ipks are valid
        condition_str = ['ideal' ,'optimal', 'normal' , 'other']
        condition = 3
        text = [] #azimuth, elevation, delta_azimuth, delta_azimuth_std, delta_elevation, delta_elevation_std,
        measured_delta_xel  =  data['delta_azimuth']* np.cos(data['elevation']) # scale due to sky shape
        abs_sky_error = measured_delta_xel
        model_delta_az, model_delta_el = ant.pointing_model.offset(data['azimuth'], data['elevation'])
        residual_az = data['delta_azimuth']   - model_delta_az
        residual_el = data['delta_elevation'] - model_delta_el
        residual_xel = residual_az * np.cos(data['elevation'])
        delta_xel_std = data['delta_azimuth_std'] * np.cos(data['elevation'])
        abs_sky_delta_std = rad2deg(np.sqrt(delta_xel_std**2 + data['delta_azimuth_std']**2))*3600 # make arc seconds
        #for i,val in enumerate(data):
        #    print ("Test Target: '%s'   fit accuracy %.3f\"  "%(data['target'][i],abs_sky_delta_std[i]))     
        abs_sky_error = rad2deg(np.sqrt((residual_xel) ** 2 + (residual_el)** 2)) *3600
    
        condition = get_condition(data)
        rms = np.std(abs_sky_error)
        robust = np.median(np.abs(abs_sky_error-abs_sky_error.mean())) * np.sqrt(2. / np.log(4.))
        text.append("Dataset:%s  Test Target: '%s' Reference RMS = %.3f\" {fit-accuracy=%.3f\"} (robust %.3f\")  (N=%i Data Points) ['%s']" % (data['dataset'][0],
            data['target'][0],rms,np.mean(abs_sky_delta_std),robust,data.shape[0],condition_str[condition]))
        
        output_data = data[0].copy() # make a copy of the rec array
        for i,x in enumerate(data[0]) :  # make an average of data 
            if x.dtype.kind == 'f' : # average floats
                output_data[i] =  data.field(i).mean()
            else : 
                output_data[i] =  data.field(i)[0]
        sun = Target('Sun,special')
        source = Target('%s,azel, %f,%f'%(output_data['target'],np.degrees(output_data['azimuth']),np.degrees(output_data['elevation'])) )
        sun_sep = np.degrees(source.separation(sun,timestamp=output_data['timestamp'],antenna=ant))  
        output_data =  recfunctions.append_fields(output_data, 'sun_sep', np.array([sun_sep]), dtypes=np.float, usemask=False, asrecarray=True)         
        output_data =  recfunctions.append_fields(output_data, 'condition', np.array([condition]), dtypes=np.float, usemask=False, asrecarray=True)
        output_data =  recfunctions.append_fields(output_data, 'rms', np.array([rms]), dtypes=np.float, usemask=False, asrecarray=True)
        output_data =  recfunctions.append_fields(output_data, 'robust', np.array([robust]), dtypes=np.float, usemask=False, asrecarray=True)
        output_data =  recfunctions.append_fields(output_data, 'N', np.array([data.shape[0]]), dtypes=np.float, usemask=False, asrecarray=True)
        
        
        
        #### Debugging
        #residual_az = data['delta_azimuth']   - model_delta_az
        #residual_el = data['delta_elevation'] - model_delta_el
        #residual_xel = residual_az * np.cos(data['elevation'])
        output_data =  recfunctions.append_fields(output_data, 'residual_az', np.array([rad2deg(residual_az.std())*3600]), dtypes=np.float, usemask=False, asrecarray=True)
        output_data =  recfunctions.append_fields(output_data, 'residual_el', np.array([rad2deg(residual_el.std())*3600]), dtypes=np.float, usemask=False, asrecarray=True)
        output_data =  recfunctions.append_fields(output_data, 'residual_xel', np.array([rad2deg(residual_xel.std())*3600]), dtypes=np.float, usemask=False, asrecarray=True)
        #print "%10s  %i  %3.1f, %s"%(data['target'][0],data['timestamp'][-1] - data['timestamp'][0], rms, str(np.degrees(data['delta_elevation']-data['delta_elevation'].mean())*3600) )
        output_data['wind_speed'] = data['wind_speed'].max()
        return text,output_data
    else :
        return None,None


def plot_source_rms(data,title):
    """Plot source pointing accuracy vs sun angles."""
    fig = plt.figure(figsize=(16,9))
    #params = {'axes.labelsize': 12, 'font.size': 10, 'legend.fontsize': 9, 
    #'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': False}
    #plt.rcParams.update(params)
    markers = []
    colors = ['b','g','r','c','m','y','k']
    pointtypes = ['o','*','x','^','s','p','h','+','D','d','v','H','d','v']
    for point in  pointtypes:
        for color in colors:
            markers.append(str(color+point))
    # sky RMS vs sun angles
    ax = fig.add_subplot(121)
    i = 0
    unique_targets = np.unique(output_data['target'])
    for target in unique_targets:
        index_list = data['target'] == target
        plt.plot(data['sun_sep'][index_list],data['rms'][index_list],markers[i],linewidth = 0, label=target)
        i = i + 1
    plt.suptitle(title, fontsize=12, fontweight='bold',y=0.95)
    plt.legend(numpoints=1,loc='upper right')
    plt.axhline(y=5,color='g',lw=1,ls='--')
    plt.axhline(y=25,color='b',lw=1,ls='--')
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Sun Angular Distance (deg)')
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # fitted Gaussian peak height vs MJD
    i = 0
    ax2 = fig.add_subplot(122)
    for target in unique_targets:
        index_list = data['target'] == target
        plt.plot((data['timestamp'][index_list]/3600.)%24,data['beam_height_I'][index_list],markers[i],linewidth = 0, label=target)
        i = i + 1
    plt.legend(numpoints=1,loc='upper right')
    plt.ylabel(r'Fitted $I_{\mathrm{peak}}$ (A.U.)')
    plt.xlabel('Time of Day (hr)')
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    return fig


def plot_diagnostics(data,title):
    """Plot offset-pointing accuracy vs environmental conditions."""
    fig = plt.figure(figsize=(16,9))
    #params = {'axes.labelsize': 12, 'font.size': 10, 'legend.fontsize': 9, 
    #'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': False}
    #plt.rcParams.update(params)
    
    colours = ['k','b', 'g', 'y']
    markers = ['o','s','^','*']
    labels = ['ideal','optimal','normal','other']
    labels_sigma = [5,5,10,25]
    
    plt.suptitle(title, fontsize=12, fontweight='bold',y=0.95)
    ax = fig.add_subplot(231)
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            plt.plot(np.degrees(data['elevation'][index_list]),data['rms'][index_list],marker=markers[i],
                color=colours[i],lw=0,label=label)
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Elevation (deg)')
    plt.legend(numpoints=1,loc='upper right')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)

    ax2 = fig.add_subplot(232)
    def time_hour(x):
        y = (x/3600.)%24 + (x/3600.)/24
        for i in xrange(x.shape[0]) :
            y = time.localtime(x[i]).tm_hour 
        return y
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            plt.plot((data['timestamp'][index_list]/3600.)%24 ,data['rms'][index_list],marker=markers[i],
                color=colours[i],lw=0,label=label)
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Hour (UTC)')
    plt.legend(numpoints=1,loc='upper right')
    plt.xlim(0,24)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)

    ax3 = fig.add_subplot(233)
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            plt.plot(data['wind_speed'][index_list],data['rms'][index_list],marker=markers[i],
                color=colours[i],lw=0,label=label)
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Wind speed (m/s)')
    plt.legend(numpoints=1,loc='upper right')
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)

    ax4 = fig.add_subplot(234)
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            plt.plot(data['sun_sep'][index_list],data['rms'][index_list],marker=markers[i],
                color=colours[i],lw=0,label=label)
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel('Sun Angular Distance (deg)')   
    plt.legend(numpoints=1,loc='upper right')
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)

    ax5 = fig.add_subplot(235)
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            plt.plot(data['temperature'][index_list],data['rms'][index_list],marker=markers[i],
                color=colours[i],lw=0,label=label)
    plt.ylabel(r'$\sigma$ (arc sec)')
    plt.xlabel(r'Temperature ($^o$C)')  
    plt.legend(numpoints=1,loc='upper right')
    ax5.yaxis.set_minor_locator(AutoMinorLocator())
    ax5.xaxis.set_minor_locator(AutoMinorLocator())
    if np.any( data['rms'] > 80 ):
        plt.ylim(0,80)

    ax6 = fig.add_subplot(236)
    for i,label in enumerate(labels):
        index_list = data['condition'] == i
        if np.sum(index_list) > 0 : 
            bin_width =  1
            plt.hist(data['rms'][index_list],bins=np.arange(0,85,bin_width),histtype='bar',ec='w',alpha=0.5,
                align='mid',color=colours[i],label=label)
    plt.legend(numpoints=1,loc='upper right')
    plt.ylabel('Number')
    plt.xlabel(r'$\sigma$ (arc sec)')
    return fig

def plots_histogram(data,title,fit=stats.rayleigh):
    """Plot offset-pointing accuracy with kde bins."""
    fig = plt.figure(figsize=(16,9))
    colours = ['k','b', 'g', 'y']
    markers = ['o','s','^','*']
    labels = ['ideal','optimal','normal','other']
    labels_sigma = [5,5,10,25] 
    gridsize = 200
    cut =  3
    bw = stats.gaussian_kde(data['rms']).scotts_factor() * data['rms'].std(ddof=1)
    try:
        import seaborn as sns
        tmp = sns.distplot(data['rms'])
    except ImportError:
        tmp = ply.hist(data['rms'],bins= np.arange(0.0, data['rms'].max() + bw * cut, bw)     )
    #print "Tmp:",tmp
    #
    plt.ylabel('Normalised Number per bin')
    plt.xlabel(r'$\sigma$ (arc sec)')
    plt.title(title)
    #gridsize = 200
    #cut =  3
    #bw = stats.gaussian_kde(data['rms']).scotts_factor() * data['rms'].std(ddof=1)
    x = np.linspace(0.0, data['rms'].max() + bw * cut, gridsize)       
    params = fit.fit(data['rms'],floc=0.0 ) # force the distrobution to start at 0.0  
    pdf = lambda x: fit.pdf(x, *params)
    y = pdf(x)
    plt.plot(x,y, label=r"Fitted a '%s' distribution with a mean = %3.3f "%(fit.name,params[1]*np.sqrt(np.pi/2.)))
    plt.legend(numpoints=1,loc='upper right')
    print "%s Fitted a '%s' distribution with mean = %3.3f "%(title,fit.name,params[1]*np.sqrt(np.pi/2.))
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


def chunk_data(data, field='target', chunk_size=5):
    """Sliding window of data sharing the same field value.

    This generator function returns a sliding window of *chunk_size*
    consecutive data elements that all share the same value for the given
    *field*. The sliding window advances one element at a time. If the field
    value changes, the current window (potentially shorter than *chunk_size*)
    is flushed.

    """
    index_list = []
    field_val = data[field][0]
    for i in xrange(len(data)):
        if data[field][i] == field_val: # the test to see if the scan is good
            if len(index_list) >= chunk_size:
                yield data[index_list]
                del(index_list[0])
            index_list.append(i) # add valid indexes
        else:
            field_val = data[field][i] # set a new value
            yield data[index_list] # flush what we have so far
            index_list = [i] # reset index list
    # Ensure that the last chunk is emitted
    if index_list:
        yield data[index_list]


def pointing_model(antenna,data):
    new_model = katpoint.PointingModel()
    num_params = len(new_model)
    default_enabled = np.array([1, 3, 4, 5, 6, 7,8]) - 1
    enabled_params = np.tile(False, num_params)
    enabled_params[default_enabled] = True
    enabled_params = enabled_params.tolist()

    # For display purposes, throw out unused parameters P2 and P10
    display_params = range(num_params)
    display_params.pop(9)
    display_params.pop(1)

    # Fit new pointing model
    az, el  =data['azimuth'],data['elevation']
    measured_delta_az, measured_delta_el = data['delta_azimuth'], data['delta_elevation']
    # Uncertainties are optional
    min_std = deg2rad((np.sqrt(2) * 60. * 1e-12)  / 60. / np.sqrt(2))
    std_delta_az = np.clip(data['delta_azimuth_std'], min_std, np.inf) \
        if 'delta_azimuth_std' in data.dtype.fields else np.tile(min_std, len(az))
    std_delta_el = np.clip(data['delta_elevation_std'], min_std, np.inf) \
        if 'delta_elevation_std' in data.dtype.fields else np.tile(min_std, len(el)) 
    params, sigma_params = new_model.fit(az, el, measured_delta_az, measured_delta_el,
                                         std_delta_az, std_delta_el, enabled_params)
    antenna.pointing_model=new_model
    return antenna
    

# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']

parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This fits a pointing model to the given data CSV file"
                               " with the targets that are included in the the offset pointing csv file ")
parser.add_option('--num-samples-limit', default=4.,
                  help="The number of valid offset measurements needed, in order to have a valid sample." )
parser.add_option('--power-sample-limit', default=7.,
                  help="The power of valid offset measurements" )
parser.add_option('--refit-pointing-model', action="store_true", default=False,
                  help="The fit a pointing model to non-selected measurements" )


parser.add_option('--no-plot', default=False,action='store_true',help="Produce a pdf output")
(opts, args) = parser.parse_args()

textString = [""]
if len(args) < 1 or not args[0].endswith('.csv'):
    raise RuntimeError('Correct File not passed to program. File should be csv file')

# read in data
data = None
for filename in args:
    if data is None:
        data,ant = read_offsetfile(filename)
    else:
        tmp_offsets,tmp_ant = read_offsetfile(filename)
        data = np.r_[data,tmp_offsets]
        if not ant == tmp_ant : raise RuntimeError('The antenna has changed')

# fix units and wraps
data['azimuth'],data['elevation']  = wrap_angle(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
data['delta_azimuth'], data['delta_elevation']= deg2rad(data['delta_azimuth']), deg2rad(data['delta_elevation'])
data['delta_azimuth_std'], data['delta_elevation_std'] = deg2rad(data['delta_azimuth_std']), deg2rad(data['delta_elevation_std'])

 
if opts.refit_pointing_model :
    ant = pointing_model(ant,data[(data['beam_height_I']  < np.float(opts.power_sample_limit))])
    print ant.pointing_model
    
output_data = None
for offsetdata in chunk_data(data):
    #New loop to provide the data in steps of test offet scans
    text,output_data_tmp = referencemetrics(ant,offsetdata,np.float(opts.num_samples_limit),np.float(opts.power_sample_limit))
    #print text#,output_data_tmp
    if not output_data_tmp is None :
        if output_data is None :
            output_data =output_data_tmp.copy()
            #print "First time"
        else:
            #print "Next time",output_data.shape[0]+1
            output_data.resize(output_data.shape[0]+1)
            output_data[-1] =output_data_tmp.copy()[0]
        textString += text

textString.append("")
textString.append(git_info() )
#for line in textString: print line
if not opts.no_plot :
    nice_filename =  args[0].split('/')[-1]+ '_residual_pointing_offset'
    if len(args) > 1 : nice_filename =  args[0].split('/')[-1]+'_Multiple_files' + '_residual_pointing_offset'
    pp = PdfPages(nice_filename+'.pdf')

    # plot diagnostic plots
    if len(output_data['condition']) > 0 :
        suptitle = '%s offset-pointing accuracy vs environmental conditions' %ant.name.upper()
        fig = plot_diagnostics(output_data,suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    
    if len(output_data['condition']) > 0 :
        suptitle = '%s offset-pointing accuracy with kde  (all sources )' %ant.name.upper()
        fig = plots_histogram(output_data,suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    
    if ((output_data['condition'] == 2) + (output_data['condition'] == 3)).sum() > 0 :
        suptitle = '%s offset-pointing accuracy with kde (normal conditions)' %ant.name.upper()
        fig = plots_histogram(output_data[ (output_data['condition'] == 2) + (output_data['condition'] == 3)],suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    if ((output_data['condition'] == 1) + (output_data['condition'] == 0)).sum() > 0 :
        suptitle = '%s offset-pointing accuracy with kde (optimal conditions)' %ant.name.upper()
        fig = plots_histogram(output_data[(output_data['condition'] == 1) + (output_data['condition'] == 0)],suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    if ((output_data['condition'] == 2) + (output_data['condition'] == 3)).sum() > 0 :
        # plot norm source RMS vs sun angles
        suptitle = '%s source-separated results (normal conditions)' %ant.name.upper()
        fig = plot_source_rms(output_data[(output_data['condition'] == 2) + (output_data['condition'] == 3)],suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)

    if ((output_data['condition'] == 1) + (output_data['condition'] == 0)).sum() > 0 :
        # plot opt source RMS vs sun angles
        suptitle = '%s source-separated results (optimal conditions)' %ant.name.upper()
        fig = plot_source_rms(output_data[(output_data['condition'] == 1) + (output_data['condition'] == 0)],suptitle)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
    
    line_step = 100    
    for lines in range(0,len(textString),line_step) :
        new_line = lines + line_step
        if new_line > len(textString) : new_line = len(textString)
        ax,fig = write_text('\n'.join(textString[lines:new_line]))
        #print '\n'.join(textString[lines:new_line] )
        fig.savefig(pp,format='pdf')
        print("Page  lines  %i to %i of %i "%(lines,new_line,len(textString)))
        old_line = lines
        plt.close(fig)
    pp.close()
