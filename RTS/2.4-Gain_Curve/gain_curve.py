#!/usr/bin/python
# Read in the results produced by analyse_point_source_scans.py
# Perform gain curve calculations and produce plots for report.
# T Mauch 24-10-2009, adapted from code originally written by S. Goedhardt

import os.path
import sys
import logging
import optparse
import glob
import time

import numpy as np
import numpy.lib.recfunctions as nprec
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
import scipy.interpolate as interpolate

from katsdpscripts.reduction.analyse_point_source_scans import batch_mode_analyse_point_source_scans
import scape
import katpoint

# These fields in the csv contain strings, while the rest of the fields are assumed to contain floats
STRING_FIELDS = ['dataset', 'target', 'timestamp_ut', 'data_unit']

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                               description="This fits gain curves to the results of analyse_point_source_scans.py")
    parser.add_option("-o", "--output", dest="outfilebase", type="string", default='./gain_curve',
                  help="Base name of output files (*.png for plots and *.csv for gain curve data)")
    parser.add_option("-p", "--polarisation", type="string", default=None, 
                  help="Polarisation to analyse, options are HH or VV. Default is all available.")
    parser.add_option("-t", "--targets", default=None, help="Comma separated list of targets to use from the input csv file. Default is all of them.")
    parser.add_option("--tsys_lim", type="float", default=150, help="Limit on calculated Tsys to flag data for atmospheric fits.")
    parser.add_option("--eff_min", type="float", default=35, help="Minimum acceptable calculated aperture efficiency.")
    parser.add_option("--eff_max", type="float", default=100, help="Maximum acceptable calculated aperture efficiency.")
    parser.add_option("--min_elevation", type="float", default=20, help="Minimum elevation to calculate statistics.")
    parser.add_option("-c", "--correct_atmosphere", action="store_true", default=False, help="Correct for atmospheric effects.")
    parser.add_option("-e", "--elev_min", type="float", default=15, help="Minimum acceptable elevation for median calculations.")
    parser.add_option("-u", "--units", default=None, help="Search for entries in the csv file with particular units. If units=counts, only compute gains. Default: first units in csv file, Options: counts, K")
    parser.add_option("-n", "--no_normalise_gain", action="store_true", default=False, help="Don't normalise the measured gains to the maximum fit to the data.")
    parser.add_option("--condition_select", type="string", default="normal", help="Flag according to atmospheric conditions (from: ideal,optimal,normal,none). Default: normal")
    parser.add_option("--csv", action="store_true", help="Input file is assumed to be csv- this overrides specified baseline")
    parser.add_option("--bline", type="string", default="sd", help="Baseline to load. Default is first single dish baseline in file")
    parser.add_option("--channel-mask", type="string", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', help="Location of rfi mask pickle file specifying channels to flag")
    parser.add_option("--ku-band", action="store_true", help="Force the center frequency of the input file to be Ku band")
    parser.add_option("--chan-range", default='211,3896', help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 211,3896)")
    (opts, args) = parser.parse_args()
    if len(args) ==0:
        print 'Please specify a file to process.'
        sys.exit(1)
    filename = args[0]
    return opts, filename

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period


def parse_csv(filename):
    """ Make an antenna object and a data array from the input csv file
    update the data array with the desired flux for the give polarisation

    Parameters
    ----------
    filename : string
        Filename containing the result of analyse_point_source_scans.py
        first line will contain the info to construct the antenna object

    Return
    ------
    :class: katpoint Antenna object
    data : heterogeneous record array
    """
    antenna = katpoint.Antenna(open(filename).readline().strip().partition('=')[2])
    #Open the csv file as an array of strings without comment fields (antenna fields are comments)
    data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
    #First non-comment line is the header with fieldnames
    fieldnames = data[0].tolist()
    #Setup all fields as float32
    formats = np.tile('float32', len(fieldnames))
    #Label the string fields as input datatype
    formats[[fieldnames.index(name) for name in STRING_FIELDS if name in fieldnames]] = data.dtype
    #Save the data as a heterogeneous record array  
    data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fieldnames, formats))
    return data, antenna


def compute_gain(data,pol):
    """ Compute the gain and apeture efficiency from the data.

    Parameters
    ----------
    data : heterogeneous record array containing 'calc_beam_height' and 'flux' records
    
    Return
    ------
    gain : The gains
    """
    gain = data['beam_height_' + pol] / data['flux']
    
    return gain


def compute_tsys_sefd(data, gain, antenna, pol):
    """ Compute Tsys and the SEFD from the gains and the baseline heights.

    Parameters
    ----------
    data : heterogeneous record array containing 'calc_temp' and 'flux' records
    gain : an array of gains calculated from the beam heights
    antenna : a katpoint:antenna object describing the antenna to use

    Return
    ------
    Tsys : The system temperature derived from the baseline heights
    SEFD : The system equivalent flux density derived from Tsys and the gain
    e    : The apeture efficiency (only meaningful if the units of 'calc_beam_height' are 'K'.)
    """
    # Get the geometric area of the dish
    ant_area = np.pi * (antenna.diameter / 2.0) ** 2
    # The apeture efficiency
    e = gain*(2761/ant_area)*100
    # Tsys can be estimated from the baseline height.
    Tsys = data['baseline_height_'+pol]
    # SEFD is Tsys/G
    SEFD = Tsys/gain

    return e, Tsys, SEFD

def select_outliers(data,pol,n_sigma=5.0):
    """ Flag data points with data['beam_height'] more than n_sigma from the median.
    Parameters
    ----------
    data : heterogeneous record array containing 'targets', 'beam_height' records
    pol : polarisation to inspect
    n_sigma : tolerance in sigma for rejecting discrepant points

    Return
    ------
    good : boolean mask of data to keep True means good data, False means bad data.
    """

    beam_heights = data['beam_height_'+pol]
    targets = np.unique(data['target'])
    elevation = data['elevation']
    good = np.ones(beam_heights.shape,dtype=np.bool)

    #Loop through targets individually
    for target in targets:
        target_indices = np.where(data['target']==target)
        beam_heights_target = beam_heights[target_indices]
        elevation_target = elevation[target_indices]
        median_beam_height = np.median(beam_heights_target)
        abs_dev = np.abs(beam_heights_target-median_beam_height)
        plt.plot(elevation_target,beam_heights_target-median_beam_height, 'ro')
        med_abs_dev=np.median(abs_dev)
        good_target = abs_dev < (1.4826*med_abs_dev)*5.0
        fit=np.polyfit(elevation_target[good_target], beam_heights_target[good_target], 1)
        abs_dev = np.abs(beam_heights_target - (fit[0]*elevation_target + fit[1]))
        med_abs_dev=np.median(abs_dev)
        good_target = abs_dev < (1.4826*med_abs_dev)*n_sigma
        good[target_indices] = good_target

    return good

def determine_good_data(data, antenna, targets=None, tsys=None, tsys_lim=150, eff=None, eff_lim=[35,100], units='K', interferometric=False, condition_select="none", pol='I'):
    """ Apply conditions to the data to choose which can be used for 
    fitting.
    Conditions are:
        1: Target name must be in 'targets' (use all targets if targets=None).
        2: Range of aperture efficiencies between eff_lim[0] and eff_lim[1].
        3: Tsys < tsys_lim.
        4: Beam height and baseline data in csv file must not be 'nan'.
        5: Units of beam height must be K

    Parameters
    ----------
    data : heterogeneous record array containing 'targets', 'beam_height' records
    targets (optional) : list of targets to keep. 'None' means use all targets.
    tsys (optional): tsys array (same lengths as data). 'None' means don't select on Tsys.
    eff (optional): array of apeture efficiencies/ 'None' means don't select on apeture efficiency.

    Return
    ------
    good : boolean mask of data to keep True means good data, False means bad data.
    """
    #Initialise boolean array of True for defaults
    good = [True] * data.shape[0]
    print "1: All data",np.sum(good)
    #Check for wanted targets
    if targets is not None:
        good = good & np.array([test_targ in targets for test_targ in data['target']])
    print "2: Flag for unwanted targets",np.sum(good)
    #Check for wanted tsys
    if tsys is not None and not interferometric:
        good = good & (tsys < tsys_lim)
    print "3: Flag for Tsys",np.sum(good)
    #Check for wanted eff
    if eff is not None and not interferometric:
        good = good & ((eff>eff_lim[0]) & (eff<eff_lim[1]))
    print "4: Flag for efficiency",np.sum(good)
    #Check for nans
    good = good & ~(np.isnan(data['beam_height_'+pol])) & ~(np.isnan(data['baseline_height_'+pol]))
    print "5: Flag for NaN in data",np.sum(good)
    #Check for units
    good = good & (data['data_unit'] == units)
    print "6: Flag for correct units",np.sum(good)
    #Check for environmental conditions if required
    if condition_select!="none":
        good = good & select_environment(data, antenna, condition_select)
    print "7: Flag for environmental condition", np.sum(good)
    #Flag discrepant gain values
    good = good & select_outliers(data,pol,5.0)
    print "8: Flag for gain outliers", np.sum(good)

    return good

def select_environment(data, antenna, condition="normal"):
    """ Flag data for environmental conditions. Options are:
    normal: Wind < 9.8m/s, -5C < Temperature < 40C, DeltaTemp < 3deg in 20 minutes
    optimal: Wind < 2.9m/s, -5C < Temperature < 35C, DeltaTemp < 2deg in 10 minutes
    ideal: Wind < 1m/s, 19C < Temp < 21C, DeltaTemp < 1deg in 30 minutes
    """
    # Convert timestamps to UTCseconds using katpoint
    timestamps = np.array([katpoint.Timestamp(timestamp) for timestamp in data["timestamp_ut"]],dtype='float32')
    # Fit a smooth function (cubic spline) in time to the temperature and wind data
    raw_wind = data["wind_speed"]
    raw_temp = data["temperature"]

    fit_wind = interpolate.InterpolatedUnivariateSpline(timestamps,raw_wind,k=3)
    fit_temp = interpolate.InterpolatedUnivariateSpline(timestamps,raw_temp,k=3)
    #fit_temp_grad = fit_temp.derivative()

    # Day/Night
    # Night is defined as when the Sun is at -5deg.
    # Set up Sun target
    sun = katpoint.Target('Sun, special',antenna=antenna)
    sun_elevation = katpoint.rad2deg(sun.azel(timestamps)[1])

    # Apply limits on environmental conditions
    good = [True] * data.shape[0]

    # Set up limits on environmental conditions
    if condition=='ideal':
        windlim        =   1.
        temp_low       =   19.
        temp_high      =   21.
        deltatemp      =   1./(30.*60.)
        sun_elev_lim   =   -5.
    elif condition=='optimum':
        windlim        =   2.9
        temp_low       =   -5.
        temp_high      =   35.
        deltatemp      =   2./(10.*60.)
        sun_elev_lim   =   -5.
    elif condition=='normal':
        windlim        =   9.8
        temp_low       =   -5.
        temp_high      =   40.
        deltatemp      =   3./(20.*60.)
        sun_elev_lim   =   100.       #Daytime
    else:
        return good

    good = good & (fit_wind(timestamps) < windlim)
    good = good & ((fit_temp(timestamps) > temp_low) & (fit_temp(timestamps) < temp_high))
    
    #Get the temperature gradient
    temp_grad = [fit_temp.derivatives(timestamp)[1] for timestamp in timestamps]
    good = good & (np.abs(temp_grad) < deltatemp)

    #Day or night?
    good = good & (sun_elevation < sun_elev_lim)

    return good



def fit_atmospheric_absorption(gain, elevation):
    """ Fit an elevation dependent atmospheric absorption model.
        Model is G=G_0*exp(-tau*airmass)

    """
    #Airmass increases as inverse sine of the elevation    
    airmass = 1/np.sin(elevation)
    #
    fit = np.polyfit(airmass, np.log(gain), 1)
    #
    tau,g_0 = -fit[0],np.exp(fit[1])

    return g_0, tau

def fit_atmospheric_emission(tsys, elevation, tau):
    """ Fit an elevation dependent atmospheric emission model.

    """
    #Airmass increases as inverse sine of the elevation    
    airmass = 1/np.sin(elevation)
    #Fit T_rec + T_atm*(1-exp(-tau*airmass))
    fit = np.polyfit(1 - np.exp(-tau*airmass),tsys,1)
    # Get T_rec and T_atm
    tatm,trec = fit[0],fit[1]

    return tatm, trec

def calc_atmospheric_opacity(T, RH, h, f):
    """
        Calculates zenith opacity according to NASA's Propagation Effects Handbook
        for Satellite Systems, chapter VI (Ippolito 1989). For elevations > 10 deg.
        Multiply by (1-exp(-opacity/sin(el))) for elevation dependence.
        Taken from katlab.
        @param T: temperature in deg C
        @param RH: relative humidity, 0 < RH < 1
        @param h: height above sea level in km
        @param f: frequency in GHz (must be < 57 GHz)
    """
    T0 = 15 # Reference temp for calculations, deg C
    # Vapour pressure
    es = 100 * 6.1121*np.exp((18.678-T0/234.5)*T0/(257.14+T0)) # [Pa], from A. L. Buck research manual 1996 rather than NASA Handbook
    rw = RH*es/(.461*(T0+273.15)) # [g/m^3]
    # Basic values
    yo = (7.19e-3+6.09/(f**2+.227)+4.81/((f-57)**2+1.50))*f**2*1e-3
    yw = (.067+3/((f-22.3)**2+7.3)+9/((f-183.3)**2+6)+4.3/((f-323.8)**2+10))*f**2*rw*1e-4
    # yw above is only for rw <= 12 g/m^3. the following alternative is suggested in NASA's handbook
    if rw>12.0:
        yw = (.05+0.0021*rw+3.6/((f-22.2)**2+8.5)+10.6/((f-183.3)**2+9)+8.9/((f-325.4)**2+26.3))*f**2*rw*1e-4
    # Correct for temperature
    yo = yo*(1-0.01*(T-T0))
    yw = yw*(1-0.006*(T-T0))
    # Scale heights
    ho = 6.
    hw = (2.2+3/((f-22.3)**2+3)+1/((f-183.3)**2+1)+1/((f-323.8)**2+1))
    # Attenuation
    A = yo*ho*np.exp(-h/ho) + yw*hw

    return np.exp(A/10.*np.log(10))-1



def make_result_report(data, good, opts, pdf, gain, e, g_0, tau, Tsys=None, SEFD=None, T_atm=None, T_rec=None):
    """ Generate a pdf report containing relevant results
        and a txt file with the plotting data.
    """
    #Set up list of separate targets for plotting
    if opts.targets:
        targets = opts.targets.split(',')
    else:
        #Plot all targets 
        targets = list(set(data['target']))
    #Separate masks for each target to plot separately
    targetmask={}
    for targ in targets:
        targetmask[targ] = np.array([test_targ==targ.strip() for test_targ in data['target']])

    #Set up range of elevations for plotting fits
    fit_elev = np.linspace(5, 90, 85, endpoint=False)
    
    #Set up the figure
    fig = plt.figure(figsize=(8.3,11.7))

    fig.subplots_adjust(hspace=0.0)
    #Plot the gain vs elevation for each target
    ax1 = plt.subplot(511)

    for targ in targets:
        # Normalise the data by fit of line to it
        if not opts.no_normalise_gain:
            use_elev = data['elevation']>opts.min_elevation
            fit_elev = data['elevation'][good & targetmask[targ] & use_elev]
            fit_gain = gain[good & targetmask[targ] & use_elev]
            fit=np.polyfit(fit_elev, fit_gain, 1)
            g90=fit[0]*90.0 + fit[1]
            if fit[0]<0.0:
                print "WARNING: Fit to gain on %s has negative slope, normalising to maximum of data"%(targ)
                g90=max(fit_gain)
            plot_gain = gain[good & targetmask[targ]]/g90
            plot_elevation = data['elevation'][good & targetmask[targ]]
            plt.plot(plot_elevation, plot_gain, 'o', label=targ)
            # Plot a pass fail line
            plt.axhline(0.95, 0.0, 90.0, ls='--', color='red')
            plt.ylabel('Normalised gain')
        else:
            plt.plot(data['elevation'][good & targetmask[targ]], gain[good & targetmask[targ]], 'o', label=targ)
            plt.ylabel('Gain (%s/Jy)'%opts.units)
    #Plot the model curve for the gains if units are K
    if opts.units!="counts":
        fit_gain = g_0*np.exp(-tau/np.sin(np.radians(fit_elev)))
        plt.plot(fit_elev, fit_gain, 'k-')

    #Get a title string
    if opts.condition_select not in ['ideal','optimum','normal']:
        condition = 'all'
    else:
        condition = opts.condition_select
    title = 'Gain Curve, '
    title += antenna.name + ','
    title += ' ' + opts.polarisation + ' polarisation,'
    title += ' ' + '%.0f MHz'%(data['frequency'][0])
    title += ' ' + '%s conditions'%(condition)
    plt.title(title)
    legend = plt.legend(loc=4)
    plt.setp(legend.get_texts(), fontsize='small')

    # Only do derived plots if units were in Kelvin
    if opts.units!="counts":
        #Plot the aperture efficiency vs elevation for each target
        ax2 = plt.subplot(512, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], e[good & targetmask[targ]], 'o', label=targ)
        plt.ylim((opts.eff_min,opts.eff_max))
        plt.ylabel('Ae  %')

        #Plot Tsys vs elevation for each target and the fit of the atmosphere
        ax3 = plt.subplot(513, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], Tsys[good & targetmask[targ]], 'o', label=targ)
        #Plot the model curve for Tsys
        fit_Tsys=T_rec + T_atm*(1 - np.exp(-tau/np.sin(np.radians(fit_elev))))
        plt.plot(fit_elev, fit_Tsys, 'k-')
        plt.ylabel('Tsys (K)')

        #Plot SEFD vs elevation for each target
        ax4 = plt.subplot(514, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], SEFD[good & targetmask[targ]], 'o', label=targ)
        plt.ylabel('SEFD (Jy)')
        xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()
        plt.setp(xticklabels, visible=False)
    

    plt.xlabel('Elevation (deg)')

    #Make some blank space for text
    ax5 = plt.subplot(515, sharex=ax1)
    plt.setp(ax5, visible=False)

    #Construct output text.
    outputtext = 'Median Gain (%s/Jy): %1.4f  std: %.4f  (el. > %2.0f deg.)\n'%(opts.units,np.median(gain[good]), np.std(gain[good]), opts.min_elevation)
    if opts.units!="counts":
        outputtext += 'Median Ae (%%):       %2.2f    std: %.2f      (el. > %2.0f deg.)\n'%(np.median(e[good]), np.std(e[good]), opts.min_elevation)
        outputtext += 'Fit of atmospheric attenuation:  '
        outputtext += 'G_0 (%s/Jy): %.4f   tau: %.4f\n'%(opts.units,g_0, tau)
    if Tsys is not None:
        outputtext += 'Median T_sys (K):   %1.2f    std: %1.2f      (el. > %2.0f deg.)\n'%(np.median(Tsys[good]),np.std(Tsys[good]),opts.min_elevation)
    if SEFD is not None:
        outputtext += 'Median SEFD (Jy):   %4.1f  std: %4.1f    (el. > %2.0f deg.)\n'%(np.median(SEFD[good]),np.std(SEFD[good]),opts.min_elevation)
    if (T_rec is not None) and (T_atm is not None):
        outputtext += 'Fit of atmospheric emission:  '
        outputtext += 'T_rec (K): %.2f   T_atm (K): %.2f'%(T_rec, T_atm)
    plt.figtext(0.1,0.1, outputtext,fontsize=11)
    fig.savefig(pdf,format='pdf')
    plt.close(fig)

#get the command line arguments
opts, filename = parse_arguments()

#No Channel mask in Ku band.
if opts.ku_band:
    opts.channel_mask=None

#Check if we're using an h5 file or a csv file and read appropriately
if opts.csv:
    # Get the data from the csv file
    data, antenna = parse_csv(filename)
else:
    #Got an h5 file - run analyse point source scans.
    file_basename = os.path.splitext(os.path.basename(filename))[0]
    prep_basename = file_basename + '_' + opts.bline.translate(None,',') + '_point_source_scans'
    antenna, data = batch_mode_analyse_point_source_scans(filename,outfilebase=os.path.abspath(prep_basename),baseline=opts.bline,
                                                            ku_band=opts.ku_band,channel_mask=opts.channel_mask,freq_chans=opts.chan_range)

if opts.units == None:
    opts.units = data['data_unit'][0]

#Get available polarisations to loop over or make a list out of options if available
if opts.polarisation == None:
    keys = np.array(data.dtype.names)
    pol = np.unique([key.split('_')[-1] for key in keys if key.split('_')[-1] in ['HH','VV','I']])
else:
    pol = opts.polarisation.split(',')

#Set up plots
# Multipage Pdf
output_filename = opts.outfilebase + '_' + antenna.name + '_' + '%.0f'%data['frequency'][0]
pdf = PdfPages(output_filename+'.pdf')

for opts.polarisation in pol:

    # Compute the gains from the data and fill the data recarray with the values
    gain = compute_gain(data,opts.polarisation)

    Tsys, SEFD, e = None, None, None
    # Get TSys, SEFD if we have meaningful units
    if opts.units=="K":
        e, Tsys, SEFD = compute_tsys_sefd(data, gain, antenna,opts.polarisation)

    # Determine "good" data to use for fitting and plotting
    good = determine_good_data(data, antenna, targets=opts.targets, tsys=Tsys, tsys_lim=opts.tsys_lim, 
                            eff=e, eff_lim=[opts.eff_min,opts.eff_max], units=opts.units,
                            condition_select=opts.condition_select, pol=opts.polarisation)

    # Check if we have flagged all the data
    if np.sum(good)==0:
        print('Pol: %s, All data flagged according to selection criteria.'%opts.polarisation)
        continue

    # Obtain desired elevations in radians
    az, el = angle_wrap(katpoint.deg2rad(data['azimuth'])), katpoint.deg2rad(data['elevation'])

    # Get a fit of an atmospheric absorption model if units are in "K", otherwise use weather data to estimate 
    # opacity for each data point
    if opts.units=="K":
        g_0, tau = fit_atmospheric_absorption(gain[good],el[good])
    else:
        tau=np.array([])
        for opacity_info in data:
            tau=np.append(tau,(calc_atmospheric_opacity(opacity_info['temperature'],opacity_info['humidity']/100, 
                                            antenna.observer.elevation/1000, opacity_info['frequency']/1000.0)))
        g_0 = None

    T_atm, T_rec = None, None
    # Fit T_atm and T_rec using atmospheric emission model for single dish case
    if opts.units=="K":
        T_atm, T_rec = fit_atmospheric_emission(Tsys[good],el[good],tau)

    #remove the effect of atmospheric attenuation from the data
    if opts.correct_atmosphere:
        if opts.units=="K":
            e = (gain -  g_0*np.exp(-tau/np.sin(el)) + g_0)*(2761/(np.pi*(antenna.diameter/2.0)**2))*100
        gain = gain/(np.exp(-tau/np.sin(el)))


    # Make a report describing the results (no Tsys data if interferometric)
    make_result_report(data, good, opts, pdf, gain, e, g_0, tau, 
                    Tsys=Tsys, SEFD=SEFD, T_atm=T_atm, T_rec=T_rec)

    #Write out gain data to file
    output_file = file(output_filename+'_'+opts.polarisation+'.csv',mode='w')
    #Header
    output_file.write("# Gain vs elevation data for %s, units of gain are: %s/Jy, Atmospheric correction?: %s\n"%(antenna.name, opts.units, opts.correct_atmosphere))
    output_file.write("#Target        ,Elev. ,  Gain  \n")
    output_file.write("# name         ,(deg.), (%s/Jy)\n"%(opts.units))
    for output_data in zip(data['target'], data['elevation'][good], gain[good]):
        output_file.write("%-15s,%4.1f  ,%7.5f\n"%(output_data[0], output_data[1],output_data[2]))

pdf.close()