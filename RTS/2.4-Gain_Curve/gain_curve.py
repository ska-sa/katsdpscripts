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
from katsdpscripts.RTS import git_info

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
                  help="Polarisation to analyse, options are I, HH or VV. Default is all available.")
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

def select_outliers(data,pol,targets,n_sigma=4.0):
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
    elevation = data['elevation']
    good = np.ones(beam_heights.shape,dtype=np.bool)

    #Loop through targets individually
    for target in targets:
        target_indices = np.where(data['target']==target)
        beam_heights_target = beam_heights[target_indices]
        elevation_target = elevation[target_indices]
        median_beam_height = np.nanmedian(beam_heights_target)
        abs_dev = np.abs(beam_heights_target-median_beam_height)
        plt.plot(elevation_target,beam_heights_target-median_beam_height, 'ro')
        med_abs_dev=np.nanmedian(abs_dev)
        good_target = abs_dev < (1.4826*med_abs_dev)*5.0
        fit=np.polyfit(elevation_target[good_target], beam_heights_target[good_target], 1)
        abs_dev = np.abs(beam_heights_target - (fit[0]*elevation_target + fit[1]))
        med_abs_dev=np.nanmedian(abs_dev)
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
    good = good & select_outliers(data,pol,targets,4.0)
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


##This is probably not necessary - use weather data to calculate tau.
##
def fit_atmospheric_absorption(gain, elevation):
    """ Fit an elevation dependent atmospheric absorption model.
        Model is G=G_0*exp(-tau*airmass)
        Assumes atmospheric conditions do no change
        over the course of the observation.
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
        Can also derive system temperature from this.
        Assumes atmospheric conditions do not change 
        over the course of the observation.
    """
    #Airmass increases as inverse sine of the elevation    
    airmass = 1/np.sin(elevation)
    #Fit T_rec + T_atm*(1-exp(-tau*airmass))
    fit = np.polyfit(1 - np.exp(-tau*airmass),tsys,1)
    # Get T_rec and T_atm
    tatm,trec = fit[0],fit[1]

    return tatm, trec

def calc_atmospheric_opacity(T, RH, P, h, f):
    """ 
        Calculates zenith opacity according to ITU-R P.676-9. For elevations > 10 deg.
        Use as "Tsky*(1-exp(-opacity/sin(el)))" for elevation dependence.
        T: temperature in deg C
        RH: relative humidity, 0 < RH < 1
        P: dry air pressure in hPa (equiv. mbar)
        h: height above sea level in km
        f: frequency in GHz (must be < 55 GHz)
        This function returns the return: approximate atmospheric opacity at zenith [Nepers]
    """
    es = 6.1121*np.exp((18.678-T/234.5)*T/(257.14+T)) # [hPa] from A. L. Buck research manual 1996
    rho = RH*es*216.7/(T+273.15) # [g/m^3] from A. L. Buck research manual 1996 (ITU-R ommited the factor "RH" - a mistake)
    
    # The following is taken directly from ITU-R P.676-9
    p_tot = P + es # from eq 3
    
    rho = rho*np.exp(h/2) # Adjust to sea level as per eq 32
    
    # eq 22
    r_t = 288./(273.+T)
    r_p = p_tot/1013.
    phi = lambda a, b, c, d: r_p**a*r_t**b*np.exp(c*(1-r_p)+d*(1-r_t))
    E_1 = phi(0.0717,-1.8132,0.0156,-1.6515)
    E_2 = phi(0.5146,-4.6368,-0.1921,-5.7416)
    E_3 = phi(0.3414,-6.5851,0.2130,-8.5854)
    # Following is valid only for f <= 54 GHz
    yo = ( 7.2*r_t**2.8 / (f**2+0.34*r_p**2*r_t**1.6) + 0.62*E_3 / ((54-f)**(1.16*E_1)+0.83*E_2) ) * f**2 * r_p**2 *1e-3
    # eq 23
    n_1 = 0.955*r_p*r_t**0.68 + 0.006*rho
    n_2 = 0.735*r_p*r_t**0.5 + 0.0353*r_t**4*rho
    g = lambda f, f_i: 1+(f-f_i)**2/(f+f_i)**2
    yw = (  3.98*n_1*np.exp(2.23*(1-r_t))/((f-22.235)**2+9.42*n_1**2)*g(f,22) + 11.96*n_1*np.exp(0.7*(1-r_t))/((f-183.31)**2+11.14*n_1**2)
          + 0.081*n_1*np.exp(6.44*(1-r_t))/((f-321.226)**2+6.29*n_1**2) + 3.66*n_1*np.exp(1.6*(1-r_t))/((f-325.153)**2+9.22*n_1**2)
          + 25.37*n_1*np.exp(1.09*(1-r_t))/(f-380)**2 + 17.4*n_1*np.exp(1.46*(1-r_t))/(f-448)**2
          + 844.6*n_1*np.exp(0.17*(1-r_t))/(f-557)**2*g(f,557) + 290*n_1*np.exp(0.41*(1-r_t))/(f-752)**2*g(f,752)
          + 8.3328e4*n_2*np.exp(0.99*(1-r_t))/(f-1780)**2*g(f,1780)
          ) * f**2*r_t**2.5*rho*1e-4
    
    # eq 25
    t_1 = 4.64/(1+0.066*r_p**-2.3) * np.exp(-((f-59.7)/(2.87+12.4*np.exp(-7.9*r_p)))**2)
    t_2 = 0.14*np.exp(2.12*r_p) / ((f-118.75)**2+0.031*np.exp(2.2*r_p))
    t_3 = 0.0114/(1+0.14*r_p**-2.6) * f * (-0.0247+0.0001*f+1.61e-6*f**2) / (1-0.0169*f+4.1e-5*f**2+3.2e-7*f**3)
    ho = 6.1/(1+0.17*r_p**-1.1)*(1+t_1+t_2+t_3)
    
    # eq 26
    sigma_w = 1.013/(1+np.exp(-8.6*(r_p-0.57)))
    hw = 1.66*( 1 + 1.39*sigma_w/((f-22.235)**2+2.56*sigma_w) + 3.37*sigma_w/((f-183.31)**2+4.69*sigma_w) + 1.58*sigma_w/((f-325.1)**2+2.89*sigma_w) )
    
    # Attenuation from dry & wet atmosphere relative to a point outside of the atmosphere
    A = yo*ho*np.exp(-h/ho) + yw*hw*np.exp(-h/hw) # [dB] from equations 27, 30 & 31
    
    return A*np.log(10)/10.0 # Convert dB to Nepers

def fit_func(x, a, b):
    return np.abs(a)*x + b

def fit_func90(x,a):
    return np.abs(a)*x 

def make_result_report_L_band(data, good, opts, pdf, gain, e,  Tsys=None, SEFD=None):
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
        targetmask[targ] = np.array([test_targ==targ.strip() for test_targ in data['target'][good]])

    #Set up range of elevations for plotting fits
    fit_elev = np.linspace(5, 90, 85, endpoint=False)
    
    obs_details = data['timestamp_ut'][0] + ', ' + data['dataset'][0]+'.h5'
    #Set up the figure
    fig = plt.figure(figsize=(8.3,11.7))

    fig.subplots_adjust(hspace=0.0, bottom=0.2, right=0.8)
    plt.suptitle(obs_details)
    
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
            #if fit[0]<0.0:
            #    print "WARNING: Fit to gain on %s has negative slope, normalising to maximum of data"%(targ)
            #    g90=max(fit_gain)
            plot_gain = gain[good & targetmask[targ]]/g90
            plot_elevation = data['elevation'][good & targetmask[targ]]
            plt.plot(plot_elevation, plot_gain, 'o', label=targ)
            # Plot a pass fail line
            plt.axhline(0.95, 0.0, 90.0, ls='--', color='red')
            plt.axhline(1.05, 0.0, 90.0, ls='--', color='red')
            plt.ylabel('Normalised gain')
        else:
            plt.plot(plot_elevation, plot_gain, 'o', label=targ)
            plt.ylabel('Gain (%s/Jy)'%opts.units)

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
    legend = plt.legend(bbox_to_anchor=(1.3, 0.7))
    plt.setp(legend.get_texts(), fontsize='small')
    plt.grid()

    # Only do derived plots if units were in Kelvin
    if opts.units!="counts":
        #Plot the aperture efficiency vs elevation for each target
        ax2 = plt.subplot(512, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], e[good & targetmask[targ]], 'o', label=targ)
        plt.ylim((opts.eff_min,opts.eff_max))
        plt.ylabel('Ae  %')
        plt.grid()

        #Plot Tsys vs elevation for each target and the fit of the atmosphere
        ax3 = plt.subplot(513, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], Tsys[good & targetmask[targ]], 'o', label=targ)
        #Plot the model curve for Tsys
        #fit_Tsys=T_rec + T_atm*(1 - np.exp(-tau/np.sin(np.radians(fit_elev))))
        #plt.plot(fit_elev, fit_Tsys, 'k-')
        plt.ylabel('Tsys (K)')
        plt.grid()

        #Plot SEFD vs elevation for each target
        ax4 = plt.subplot(514, sharex=ax1)
        for targ in targets:
            plt.plot(data['elevation'][good & targetmask[targ]], SEFD[good & targetmask[targ]], 'o', label=targ)
        plt.ylabel('SEFD (Jy)')
        xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.grid()
    

    plt.xlabel('Elevation (deg)')

    #Make some blank space for text
    ax5 = plt.subplot(515, sharex=ax1)
    plt.setp(ax5, visible=False)

    #Construct output text.
    outputtext = 'Median Gain (%s/Jy): %1.4f  std: %.4f  (el. > %2.0f deg.)\n'%(opts.units,np.median(gain[good]), np.std(gain[good]), opts.min_elevation)
    if opts.units!="counts":
        outputtext += 'Median Ae (%%):       %2.2f    std: %.2f      (el. > %2.0f deg.)\n'%(np.median(e[good]), np.std(e[good]), opts.min_elevation)
    if Tsys is not None:
        outputtext += 'Median T_sys (K):   %1.2f    std: %1.2f      (el. > %2.0f deg.)\n'%(np.median(Tsys[good]),np.std(Tsys[good]),opts.min_elevation)
    if SEFD is not None:
        outputtext += 'Median SEFD (Jy):   %4.1f  std: %4.1f    (el. > %2.0f deg.)\n'%(np.median(SEFD[good]),np.std(SEFD[good]),opts.min_elevation)
    plt.figtext(0.1,0.1, outputtext,fontsize=11)
    plt.figtext(0.89, 0.09, git_info(), horizontalalignment='right',fontsize=10)
    fig.savefig(pdf,format='pdf')
    plt.close(fig)

def scale_gain(g, nu_0, nu, el):
    """ 
    Scale gain to higher frequency using scaling law of Ruze equation
    Returns predicted gain over elevation range for plotting purposes,
    and predicted gain at 15 and 90 degree elevation per SE requirement
    """
    scale = (nu**2/nu_0**2)-1
    g_el = g(el-90.) + 1.
    g_15 = g(15-90.) + 1.
    g_90 = g(0.) + 1.
    return (g_el**scale)*g_el, (g_15**scale)*g_15, (g_90**scale)*g_90

def parabolic_func(x,a,b,c):
    """
    Return the y value of a parabola at x
    """
    return a*(x-b)**2 + c

def fit_parabola(data_x,data_y,pos=60):
    """
    Fit a parabola to multiple datasets where the height can vary for each datset 
    but the shape is fitted across all datsets. The position of the peak is held constant. 
    """
    def chi_squared(x,y,a,b,c):
        """
        Get chi-squared for a set of data points, x,y for parabola with parameter a,b,c
        """
        return np.sum((y-parabolic_func(x,a,b,c))**2)

    def residual(deps,data_x,data_y,pos):
        """
        Residual function, with different height parameter for each datset in (2d) arrays data_x and data_y
        """
        shape=deps[0]
        height=deps[1:]
        total_residual=0.0
        for num,this_height in enumerate(height):
            #print num,chi_squared(data_x[num],data_y[num],shape,pos,this_height)
            total_residual+=chi_squared(data_x[num],data_y[num],shape,pos,this_height)
        return total_residual

    height_guess=[np.mean(y) for y in data_y]
    init_guess=[-0.00001] + height_guess
    bounds_height =[(0.0,None) for y in data_y]
    bounds_all = [(None,0.0)] + bounds_height
    test=optimize.minimize(residual,init_guess,(data_x,data_y,pos,),bounds=bounds_all)
    return test
        
def make_result_report_ku_band(gain, opts, targets, pdf):
    """
       No noise diode present at ku-band.  
       Gains will always have to be normalised.  
       We are interested in the relative gain change between 15 to 90 degrees elevation
    
    """
    #Separate masks for each target to plot separately
    targetmask={}
    for targ in targets:
        targetmask[targ] = np.array([test_targ==targ.strip() for test_targ in data['target']])
    #Set up range of elevations for plotting fits
    fit_elev = np.linspace(5, 90, 85, endpoint=False)
    
    obs_details = data['timestamp_ut'][0] + ', ' + data['dataset'][0]+'.h5'
    #Set up the figure
    fig = plt.figure(figsize=(8.3,11.7))

    fig.subplots_adjust(hspace=0.0, bottom=0.2)
    plt.suptitle(obs_details)
    
    #Plot the gain vs elevation for each target
    ax1 = plt.subplot(511)
    #get ready to collect normalised gains for each target.
    norm_gain = list()
    norm_elev = list()
    all_elev=[]
    all_gain=[]
    for targ in targets:
        # Normalise the data by fit of line to it
        if not opts.no_normalise_gain:
            use_elev = data['elevation']>opts.min_elevation
            fit_elev = data['elevation'][good & targetmask[targ] & use_elev]
            fit_gain = gain[good & targetmask[targ] & use_elev]
            all_elev.append(fit_elev)
            all_gain.append(fit_gain)
    test=fit_parabola(all_elev,all_gain)
    for n,dat in enumerate(zip(all_elev,all_gain)):
        plt.plot(dat[0],dat[1]/test['x'][n+1],'.')
    print test
    print np.std(dat[1]/test['x'][n+1])
    plt.plot(np.arange(0,90,1),parabolic_func(np.arange(0,90,1),test['x'][0],60,1))
    plt.show()
    sys.exit()
    fit, cov = optimize.curve_fit(fit_func,fit_elev, fit_gain)
    g90=np.abs(fit[0])*90.0 + fit[1]
    #if fit[0]<0.0:
    #    print "WARNING: Fit to gain on %s has negative slope, normalising to maximum of data"%(targ)
    #    g90=max(fit_gain)
    plot_gain = gain[good & targetmask[targ]]/g90
    plot_elevation = data['elevation'][good & targetmask[targ]]
    plt.plot(plot_elevation, plot_gain, 'o', label=targ)
    norm_gain.append(plot_gain)
    norm_elev.append(plot_elevation)
    plt.ylabel('Normalised gain')
    plt.xlabel('Elevation (deg)')
    norm_gain_90 = np.hstack(norm_gain) - 1.
    norm_elev_90 = np.hstack(norm_elev) - 90.

    fit, cov = optimize.curve_fit(fit_func90, norm_elev_90, norm_gain_90)
    fit[0] = np.abs(fit[0])
    g = np.poly1d([fit[0],0.])
    fit_elev = np.linspace(20, 90, 85, endpoint=False)
    plt.plot(fit_elev, g(fit_elev-90.)+1., label='12.5 GHz fit')

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
    plt.grid()
    
    nu = 14.5e9
    nu_0 = 12.5e9
    g14, g14_15, g14_90 = scale_gain(g, nu_0, nu, fit_elev)
    loss_12 = (g(0.) - g(15.-90.))/(g(0.)+1.)*100
    loss_14 = (g14_90 - g14_15)/g14_90*100
    
    plt.plot(fit_elev, g14, label = '14.5 GHz fit')
    legend = plt.legend(bbox_to_anchor=(1, -0.1))
    plt.setp(legend.get_texts(), fontsize='small')
    outputtext = 'Relative loss in gain at 12.5 GHz is %.2f %%\n'%loss_12
    outputtext +='Relative loss in gain at 14.5 GHz is %.2f %%'%loss_14
    plt.figtext(0.1,0.55, outputtext,fontsize=11)
    plt.figtext(0.89, 0.5, git_info(), horizontalalignment='right',fontsize=10)
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
    file_basename = data['dataset'][0]
else:
    #Got an h5 file - run analyse point source scans.
    file_basename = os.path.splitext(os.path.basename(filename))[0]
    prep_basename = file_basename + '_' + opts.bline.translate(None,',') + '_point_source_scans'
    antenna, data = batch_mode_analyse_point_source_scans(filename,outfilebase=os.path.abspath(prep_basename),baseline=opts.bline,
                                                            ku_band=opts.ku_band,channel_mask=opts.channel_mask,freq_chans=opts.chan_range,remove_spikes=True)
#Check we've some data to process
if len(data['data_unit'])==0:
    sys.exit()

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
output_filename = opts.outfilebase + '_' + file_basename + '_' + antenna.name + '_' + '%.0f'%data['frequency'][0]
pdf = PdfPages(output_filename+'.pdf')

for opts.polarisation in pol:

    # Compute the gains from the data and fill the data recarray with the values
    gain = compute_gain(data,opts.polarisation)

    Tsys, SEFD, e = None, None, None
    # Get TSys, SEFD if we have meaningful units
    if opts.units=="K":
        e, Tsys, SEFD = compute_tsys_sefd(data, gain, antenna,opts.polarisation)

    targets = opts.targets.split(',') if opts.targets else np.unique(data['target'])
    targets = [target for target in targets if np.sum(data['target']==target)>1]
    # Determine "good" data to use for fitting and plotting
    good = determine_good_data(data, antenna, targets=targets, tsys=Tsys, tsys_lim=opts.tsys_lim, 
                            eff=e, eff_lim=[opts.eff_min,opts.eff_max], units=opts.units,
                            condition_select=opts.condition_select, pol=opts.polarisation)

    # Check if we have flagged all the data
    if np.sum(good)==0:
        print('Pol: %s, All data flagged according to selection criteria.'%opts.polarisation)
        continue

    # Obtain desired elevations in radians
    az, el = angle_wrap(katpoint.deg2rad(data['azimuth'])), katpoint.deg2rad(data['elevation'])

    #Correct for atmospheric opacity
    if opts.correct_atmosphere:
        tau=np.array([])
        for opacity_info in data:
            tau = np.append(tau,(calc_atmospheric_opacity(opacity_info['temperature'],opacity_info['humidity']/100, 
                                opacity_info['pressure'], antenna.observer.elevation/1000, opacity_info['frequency']/1000.0)))
        gain = (gain/(np.exp(-tau/np.sin(el)))).astype(np.float32)

    # Make a report describing the results (no Tsys data if interferometric)
    if opts.ku_band:
        make_result_report_ku_band(gain, opts, targets, pdf)
    else:
        make_result_report_L_band(data, good, opts, pdf, gain, e, Tsys, SEFD)

    #Write out gain data to file
    output_file = file(output_filename+'_'+opts.polarisation+'.csv',mode='w')
    #Header
    output_file.write("# Gain vs elevation data for %s, units of gain are: %s/Jy, Atmospheric correction?: %s\n"%(antenna.name, opts.units, opts.correct_atmosphere))
    output_file.write("#Target        ,Elev. ,  Gain  \n")
    output_file.write("# name         ,(deg.), (%s/Jy)\n"%(opts.units))
    for output_data in zip(data['target'], data['elevation'][good], gain[good]):
        output_file.write("%-15s,%4.1f  ,%7.5f\n"%(output_data[0], output_data[1],output_data[2]))

pdf.close()
