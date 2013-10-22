#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
# Reduces tipping curve data and fits a tipping model to the data
# Needs: noise diode models and tipping spillover model
# The script expects to find the following files in a specified directory
# 'K7H_1200.txt', 'K7V_1200.txt', 'K7H_1600.txt', 'K7V_1600.txt', 'K7H_2000.txt', 'K7V_2000.txt'
# If the filenames or frequencies change, please modify the function interp_spillover accordingly
# Note also that this version takes into account of the Tsky by reading a sky model from  my /usr/local/aips/FITS/TBGAL_CONVL.FITS
# You also need pyfits. The TBGAL_CONVL is a map at 1.4GHz convolved with 1deg resolution
# To run type the following: %run fit_tipping_curve_nad.py -a 'A7A7' -t  /Users/nadeem/Dev/svnScience/KAT-7/comm/scripts/K7_tip_predictions
# /mrt2/KAT/DATA/Tipping/Ant7/1300572919.h5
#
import sys
import optparse
import re
import os.path
import logging
import numpy as np
import matplotlib.pyplot as plt
import pyfits
import scipy

import scape
from katpoint import rad2deg, deg2rad,  construct_azel_target
from math import *

class Sky_temp:
    """
       T_sky = T_cont + T_casA + T_HI + T_cmb
       Read in the convolved file, and provide a method of passing back the Tsky temp a at a position
    """
    def __init__(self,inputfile='~/comm/catalogues/TBGAL_CONVL.FITS',nu=1828.0):
        """ Load The data from inputfile """
        self.ra =  lambda x: int(x/0.25) # helper functions
        self.dec = lambda x: int((-x+90)/0.25)
        self.nu = nu
        self.alpha = -0.727
        def Tsky_approx(ra,dec):
            T_cmb = 2.7
            T_gal = 10.0 * (self.nu/408) ** (-(2-self.alpha))
            return T_cmb + T_gal
        self.Tsky_approx = Tsky_approx
        try:
            hdulist = pyfits.open(inputfile)
            self.Data = np.flipud(np.fliplr(hdulist[0].data)) # data is in the first element of the fits file
            self.Data_imshow = np.flipud(hdulist[0].data) # data is in the first element of the fits file
            def Tsky(ra,dec):
                return self.Data[self.dec(dec),self.ra(ra)]*(self.nu/1420.0)**(-(2-self.alpha))
            self.data_freq = 1420.0
            self.Tsky =  Tsky
        except IOError:
            logger.error('Failed to load sky tempreture map using approximations')
            self.Tsky =  self.Tsky_approx
            Data = np.zeros([360,180])
            for ra in range(360):
                for dec in range(-90,90):
                    Data[ra,dec+90] =  self.Tsky(ra,dec)
            self.Data = Data
            self.data_freq = self.nu

    def set_freq(self,nu):
        """ Set the frequency. This is only needed for approximations """
        self.nu = nu

    def plot_sky(self,ra=None,dec=None,figure_no=9):
        plt.figure(figure_no)
        plt.clf()
        if not ra is None :
            Ra_obs = ra
            Dec_obs = dec
            plt.plot(Ra_obs,Dec_obs,'ro')
        plt.xlabel("RA(J2000) [degrees]")
        plt.ylabel("Dec(J2000) [degrees]")
        plt.imshow(np.fliplr(self.Data)*(self.nu/self.data_freq)**(-(2-self.alpha)), extent=[360,0,-90,90],vmax=50)

class Spill_Temp:
    """Load spillover models and interpolate to centre observing frequency."""
    def __init__(self,nu,path=''):
        spillover_1200_HH_file = os.path.join(path, 'K7H_1200.txt')
        spillover_1200_VV_file = os.path.join(path, 'K7V_1200.txt')
        spillover_1600_HH_file = os.path.join(path, 'K7H_1600.txt')
        spillover_1600_VV_file = os.path.join(path, 'K7V_1600.txt')
        spillover_2000_HH_file = os.path.join(path, 'K7H_2000.txt')
        spillover_2000_VV_file = os.path.join(path, 'K7V_2000.txt')
        spillover_1200H = np.loadtxt(spillover_1200_HH_file, unpack=True)
        spillover_1600H = np.loadtxt(spillover_1600_HH_file, unpack=True)
        spillover_2000H = np.loadtxt(spillover_2000_HH_file, unpack=True)
        spillover_1200V = np.loadtxt(spillover_1200_VV_file, unpack=True)
        spillover_1600V = np.loadtxt(spillover_1600_VV_file, unpack=True)
        spillover_2000V = np.loadtxt(spillover_2000_VV_file, unpack=True)
        # Extract elevations (assumes same elevation values for all models
        # Provided models are a function of zenith angle
        elH = 90. - spillover_2000H[0]
        sort_ind = elH.argsort()
        elH = elH[sort_ind]
        elV = 90. - spillover_2000V[0]
        sort_ind = elV.argsort()
        elV = elV[sort_ind]
        if (nu >= 1600) and (nu <= 2000):
            spilloverH_nu = (nu-1600)/(2000-1600)*spillover_2000H[1] + (1 - (nu-1600)/(2000-1600))*spillover_1600H[1]
            spilloverV_nu = (nu-1600)/(2000-1600)*spillover_2000V[1] + (1 - (nu-1600)/(2000-1600))*spillover_1600V[1]
        elif (nu >= 1200) and (nu <= 1600):
            spilloverH_nu = (nu-1200)/(1600-1200)*spillover_1600H[1] + (1 - (nu-1200)/(1600-1200))*spillover_1200H[1]
            spilloverV_nu = (nu-1200)/(1600-1200)*spillover_1600V[1] + (1 - (nu-1200)/(1600-1200))*spillover_1200V[1]
        T_HH = scape.fitting.PiecewisePolynomial1DFit()
        T_VV = scape.fitting.PiecewisePolynomial1DFit()
        T_HH.fit(elH, spilloverH_nu[sort_ind])
        T_VV.fit(elV, spilloverV_nu[sort_ind])
        self.spill = {}
        self.spill['HH'] = T_HH
        self.spill['VV'] = T_VV

class System_Temp:
    """Extract tipping curve data points and surface temperature."""
    def __init__(self,d,path='~/comm/catalogues/TBGAL_CONVL.FITS'):#d, nu, pol
        """ First extract total power in each scan (both mean and standard deviation) """
        T_skytemp = Sky_temp(inputfile=path)
        T_skytemp.set_freq( d.freqs[0])
        T_sky =  T_skytemp.Tsky
        self.units = d.data_unit
        self.name = d.antenna.name
        self.filename = args[0]
        self.elevation =  {}
        self.Tsys = {}
        self.sigma_Tsys = {}
        self.Tsys_sky = {}
        self.T_sky = []
        # Sort data in the order of ascending elevation
        elevation = np.array([np.average(scan_el) for scan_el in scape.extract_scan_data(d.scans,'el').data])
        ra        = np.array([np.average(scan_ra) for scan_ra in scape.extract_scan_data(d.scans,'ra').data])
        dec       = np.array([np.average(scan_dec) for scan_dec in scape.extract_scan_data(d.scans,'dec').data])
        sort_ind = elevation.argsort()
        elevation,ra,dec = elevation[sort_ind],ra[sort_ind],dec[sort_ind]
        valid_el = (elevation >= 10)
        self.elevation =  elevation[valid_el]
        self.ra = ra[valid_el]
        self.dec = dec[valid_el]
        self.surface_temperature = np.mean(d.enviro['temperature']['value'])# Extract surface temperature from weather data
        self.freq = d.freqs[0]  #MHz Centre frequency of observation
        for pol in ['HH','VV']:
            power_stats = [scape.stats.mu_sigma(s.pol(pol)[:, 0]) for s in d.scans]
            tipping_mu, tipping_sigma = np.array([s[0] for s in power_stats]), np.array([s[1] for s in power_stats])
            tipping_mu, tipping_sigma = tipping_mu[sort_ind], tipping_sigma[sort_ind]
            self.Tsys[pol] = tipping_mu[valid_el]
            self.sigma_Tsys[pol] = tipping_sigma[valid_el]
            self.Tsys_sky[pol] = []
            self.T_sky = []
            for val_el,ra,dec,el in zip(sort_ind,self.ra,self.dec,self.elevation):
                self.T_sky.append( T_sky(ra,dec))
                self.Tsys_sky[pol].append(tipping_mu[val_el]-T_sky(ra,dec))
        TmpSky = scape.fitting.PiecewisePolynomial1DFit()
        TmpSky.fit(self.elevation, self.T_sky)
        self.Tsky = TmpSky
        T_skytemp.plot_sky(self.ra,self.dec)


    def __iter__(self):
        return self

    def next(self):
        i = -1
        while True:
            i = i + 1
            if not self.ra[i]:raise StopIteration
            yield i,self.ra[i],self.dec[i],self.elevation[i]

def load_cal(filename, baseline, start_freq_channel, end_freq_channel, nd_models):
    """ Load the dataset into memory """
    d = scape.DataSet(filename, baseline=baseline, nd_models=nd_models)
    d = d.select(freqkeep=range(start_freq_channel, end_freq_channel + 1))
    d = d.convert_power_to_temperature(min_duration=opts.min_nd, jump_significance=10.0)
    d = d.select(flagkeep='~nd_on')
    d = d.select(labelkeep='track', copy=False)
    d.average()
    return d



def fit_tipping(T_sys,SpillOver,pol):
    """Fit tipping curve.
        T_sys(el) = T_cmb + T_gal + T_atm*(1-exp(-tau_0/sin(el))) + T_spill(el) + T_rx
        We will fit the opacity and T_rx """
    T_atm = 1.12 * (273.15 + T_sys.surface_temperature) - 50.0 # ??
    # Create a function to give the spillover at any elevation at the observing frequency
    # Set up full tipping equation y = f(p, x):
    #   function input x = elevation in degrees
    #   parameter vector p = [T_rx, zenith opacity tau_0]
    #   function output y = T_sys in kelvin
    #   func = lambda p, x: p[0] + T_cmb + T_gal + T_spill_func(x) + T_atm * (1 - np.exp(-p[1] / np.sin(deg2rad(x))))
    #T_sky = np.average(T_sys.T_sky)# T_sys.Tsky(x)
    func = lambda p, x: p[0] +  T_sys.Tsky(x) + SpillOver.spill[pol](x) + T_atm * (1 - np.exp(-p[1] / np.sin(deg2rad(x))))
    # Initialise the fitter with the function and an initial guess of the parameter values
    tip = scape.fitting.NonLinearLeastSquaresFit(func, [70, 0.005])
    tip.fit(T_sys.elevation, T_sys.Tsys[pol])
    logger.info('Fit results for %s polarisation:' % (pol,))
    logger.info('T_ant = %.2f K' % (tip.params[0],))
    logger.info('Zenith opacity tau_0 = %.5f' % (tip.params[1],))
    # Calculate atmospheric noise contribution at 10 degrees elevation for comparison with requirements
    T_atm_10 = T_atm * (1 - np.exp(-tip.params[1] / np.sin(deg2rad(10))))
    fit_func = []
    logger.info('Atmospheric noise contribution at 10 degrees is: %.2f K' % (T_atm_10,))
    for el in T_sys.elevation: fit_func.append(func(tip.params,el))
    return {'params': tip.params,'fit':fit_func,'scatter': (T_sys.Tsys[pol]-fit_func),'chisq':chisq_pear(fit_func,T_sys.Tsys[pol])}

def plot_data(Tsys,fit_HH,fit_VV):
    fig = plt.figure(10)
    plt.clf()
    F=plt.gcf()
    textsize = 9
    left, width = 0.1, 0.8
    rect1 = [left, 0.5, width, 0.4]
    rect2 = [left, 0.27, width, 0.15]
    rect3 = [left, 0.1, width, 0.15]
    ax1 = fig.add_axes(rect1)
    plt.plot(Tsys.elevation, Tsys.Tsys['HH'], marker='o', color='b', label='HH measured', linewidth=0)
    plt.errorbar(Tsys.elevation, Tsys.Tsys['HH'], Tsys.sigma_Tsys['HH'], ecolor='b', color='b', capsize=6, linewidth=0)
    plt.plot(Tsys.elevation, fit_HH['fit'], color='b' , label="HH $T_{rec}$ = %.2f K, $T_0$ = %.5f \n HH-$X^{2}$= %.3f" % (fit_HH['params'][0], fit_HH['params'][1], fit_HH['chisq']))
    plt.plot(Tsys.elevation, Tsys.Tsys['VV'], marker='^', color='r', label='VV measured', linewidth=0)
    plt.errorbar(Tsys.elevation, Tsys.Tsys['VV'],  Tsys.sigma_Tsys['VV'], ecolor='r', color='r', capsize=6, linewidth=0)
    plt.plot(Tsys.elevation, fit_VV['fit'], color='r', label="VV $T_{rec}$ = %.2f K, $T_0$ = %.5f \n VV-$X^{2}$= %.3f" %  (fit_VV['params'][0], fit_VV['params'][1], fit_VV['chisq']))
    plt.legend()
    plt.title('Tipping curve for antenna %s using data file: %s' % (Tsys.name,Tsys.filename))
    plt.xlabel('Elevation (degrees)')
    plt.grid()
    if Tsys.units == 'K':
        plt.ylabel('Tsys (K)')
    else:
        plt.ylabel('Raw power (counts)')
        plt.legend()

    ax2 = fig.add_axes(rect2, sharex=ax1)
    plt.ylim(np.floor(min(fit_HH['scatter'])),np.ceil(max(fit_HH['scatter'])))
    markerline, stemlines, baseline = plt.stem(Tsys.elevation, fit_HH['scatter'], '-.', markerfmt='bo')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    plt.legend(("","HH"))
    plt.grid()
    ax3  = fig.add_axes(rect3, sharex=ax1)
    plt.ylim(np.floor(min(fit_VV['scatter'])),np.ceil(max(fit_VV['scatter'])))
    markerline, stemlines, baseline = plt.stem(Tsys.elevation, fit_VV['scatter'], '-.', markerfmt='r^')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    plt.legend(("","VV"))
    plt.grid()
    plt.show()


def chisq_pear(fit,Tsys):
    chisq=0.0
    scatter = Tsys-fit
    for i in range(len(scatter)):
        chisq_pearson=(scatter[i]**2/fit[i])
        chisq=(scatter[i]**2/np.var(Tsys))
        chisq_pearson+=chisq_pearson # will use later to check
        chisq+=chisq
    return chisq


# Set up logging: log everything (DEBUG & above)
logger = logging.root
logger.setLevel(logging.DEBUG)

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This reduces a data file to produce a tipping curve plot.')
parser.add_option("-a", "--baseline", default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-n", "--nd-models",
                  help="Name of optional directory containing noise diode model files")
parser.add_option("-m", "--min-nd", type="float", default=3.0,
                  help="Minimum duration of noise diode on/off segments to use for calibration, in seconds (default = %default)")
parser.add_option("-f", "--freq-chans", default='100,400',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option("-t", "--tip_models",
                  help="Name of directory containing spillover models (**required**)")
parser.add_option("-o", "--output", dest="outfilebase", default='spectra',
                  help="Base name of output files (*.png for figures), default = '%default'")
parser.add_option( "--sky-map", default='~/comm/catalogues/TBGAL_CONVL.FITS',
                  help="Name of map of sky tempreture in fits format', default = '%default'")

(opts, args) = parser.parse_args()


if len(args) < 1:
    logger.error('Please specify the data file to reduce')
    sys.exit(1)

if not opts.tip_models:
    logger.error('Please specify the directory containing spillover models')
    sys.exit(1)

#Load the data file
d = load_cal(args[0], opts.baseline, int(opts.freq_chans.split(',')[0]),int(opts.freq_chans.split(',')[1]), opts.nd_models)
nu = d.freqs[0]  #MHz Centre frequency of observation
SpillOver = Spill_Temp(nu,path=opts.tip_models)
T_SysTemp = System_Temp(d,opts.sky_map)

fit_HH = fit_tipping(T_SysTemp,SpillOver,'HH')
fit_VV = fit_tipping(T_SysTemp,SpillOver,'VV')

logger.info('Chi square for HH is: %.6f ' % (fit_HH['chisq'],))
logger.info('Chi square for VV is: %.6f ' % (fit_VV['chisq'],))

plot_data(T_SysTemp,fit_HH,fit_VV)
#
# Save option to be added
#
save =  raw_input('Press s to save files, enter to continue: ')
if save=='s' or save=='S':
    outfilebase = args[0]+ '_' +d.antenna.name+'_tipping'
    plt.figure(10)
    plt.savefig(outfilebase+'.png', dpi=100)
    outfilebase = args[0]+ '_' +d.antenna.name+'_sky'
    plt.figure(9)
    plt.savefig(outfilebase+'.png', dpi=100)
