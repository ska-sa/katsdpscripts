#!/usr/bin/python
# Reduces tipping curve data and fits a tipping model to the data
# Needs: noise diode models and tipping spillover model
# The script expects to find the following files in a specified directory
# 'K7H_1200.txt', 'K7V_1200.txt', 'K7H_1600.txt', 'K7V_1600.txt', 'K7H_2000.txt', 'K7V_2000.txt'
# If the filenames or frequencies change, please modify the function interp_spillover accordingly
# Note also that this version takes into account of the Tsky by reading a sky model from  my TBGAL_CONVL.FITS
# You also need pyfits. The TBGAL_CONVL is a map at 1.4GHz convolved with 1deg resolution
# To run type the following: %run fit_tipping_curve_nad.py -a 'A7A7' -t  /Users/nadeem/Dev/svnScience/KAT-7/comm/scripts/K7_tip_predictions
# /mrt2/KAT/DATA/Tipping/Ant7/1300572919.h5
#
import sys
import optparse
import re
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pyfits

import warnings
from matplotlib.backends.backend_pdf import PdfPages
import katfile
import scape
import scikits.fitting as fit
from katpoint import rad2deg, deg2rad,  construct_azel_target


class Sky_temp:
    """
       T_sky = T_cont + T_casA + T_HI + T_cmb
       Read in the convolved file, and provide a method of passing back the Tsky temp a at a position
    """
    def __init__(self,inputfile='TBGAL_CONVL.FITS',nu=1828.0):
        """ Load The Tsky data from an inputfile in FITS format and scale to frequency
        This takes in 2 optional parameters:
        inputfile (filename) a fits file at 1420 MHz
        nu (MHz) center frequency 
        the data is scaled by the alpha=-0.727
        This needs to be checked
        This initilises the sky temp object. 

        """
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
            warnings.warn('Warning: Failed to load sky tempreture map using approximations')
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

    def plot_sky(self,ra=None,dec=None,figure_no=None):
        """ plot_sky plots the sky tempreture and overlays pointing centers as red dots
        The sky tempreture is the data that was loaded when the class was iniitated.
        plot_sky takes in 3 optional parameters:
                ra,dec  are list/1D-array like values of right assension and declanation
                figure_no is the figure number, None just makes a new figure.
        returns matplotlib figure object that the plot is assosated with.
        """
        if figure_no is None:
             fig = plt.figure()
        else:
            fig =plt.figure(figure_no)
            fig.clf()
        if not ra is None and not dec is None :
            if len(dec) == len(ra) :
                plt.plot(ra,dec,'ro')
        else:
            raise RuntimeError('Number of Declanation values (%s) is not equal to the number of Right assension values (%s) in plot_sky'%(len(dec),len(ra)))
        plt.xlabel("RA(J2000) [degrees]")
        plt.ylabel("Dec(J2000) [degrees]")
        plt.imshow(np.fliplr(self.Data), extent=[360,0,-90,90],vmax=50) #*(self.nu/self.data_freq)**(-(2-self.alpha))
        return fig

class Spill_Temp:
    """Load spillover models and interpolate to centre observing frequency."""
    def __init__(self,path=''):
        """ The class Spill_temp reads the spillover model from file and
        produces fitted functions for a frequency
        The class/__init__function takes in one parameter:
        path : (default='') This is the path to the directory containing
               spillover models that have names
               kat7H_spill.txt,kat7V_spill.txt
               these files have 3 cols:
                theta(Degrees, 0 at Zenith),tempreture (MHz),Frequency (MHz)
               if there are no files zero spilover is assumed.
        returns :
               dict  spill with two elements 'HH' 'VV' that
               are intepolation functions that take in elevation & Frequency(MHz)
               and return tempreture in Kelven.
        """
#TODO Need to sort out better frequency interpolation & example
        try:
            spillover_H_file = os.path.join(path, 'kat7H_spill.txt')
            spillover_V_file = os.path.join(path, 'kat7V_spill.txt')
            spillover_H = np.loadtxt(spillover_H_file, unpack=True)
            spillover_V = np.loadtxt(spillover_V_file, unpack=True)
        except IOError:
            spillover_H = np.array([[0.0,90.0,0.0,90.0],[0.,0.,0.,0.],[1200.,1200.0,2000,2000]])
            spillover_V = np.array([[0.0,90.0,0.0,90.0],[0.,0.,0.,0.],[1200.,1200.0,2000,2000]])
            warnings.warn('Warning: Failed to load Spillover models, setting models to zeros')
        # the models are in a format of theta=0  == el=90
        spillover_H[0]= 90-spillover_H[0]
        spillover_V[0]= 90-spillover_V[0]
        
        #np.array([[33.0,],[1200.]])
        #Assume  Provided models are a function of zenith angle & frequency
        T_H = fit.Spline2DScatterFit(degree=(1,1))
        T_V = fit.Spline2DScatterFit(degree=(1,1))
        T_H.fit(spillover_H[[0,2],:],spillover_H[1,:])
        T_V.fit(spillover_V[[0,2],:],spillover_V[1,:])
        self.spill = {}
        self.spill['HH'] = T_H # The HH and VV is a scape thing
        self.spill['VV'] = T_V
        self.spillfig = plt.figure()
        plt.title('Spillover models')
        

class System_Temp:
    """Extract tipping curve data points and surface temperature."""
    def __init__(self,d,path='TBGAL_CONVL.FITS',freqs=1822):#d, nu, pol
        """ First extract total power in each scan (both mean and standard deviation) """
        T_skytemp = Sky_temp(inputfile=path,nu=freqs)
        T_skytemp.set_freq(freqs)
        print freqs
        T_sky =  T_skytemp.Tsky
        self.units = d.data_unit
        self.name = d.antenna.name
        self.filename = d.filename
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
        self.sky_fig = T_skytemp.plot_sky(self.ra,self.dec)


    def __iter__(self):
        return self

    def next(self):
        i = -1
        while True:
            i = i + 1
            if not self.ra[i]:raise StopIteration
            yield i,self.ra[i],self.dec[i],self.elevation[i]

def load_cal(filename, baseline, start_freq_channel, end_freq_channel):
    """ Load the dataset into memory """
    d = scape.DataSet(filename, baseline=baseline)#, nd_models=nd_models
    d = d.select(freqkeep=range(start_freq_channel, end_freq_channel + 1))
    d = d.convert_power_to_temperature()
    d = d.select(flagkeep='~nd_on')
    d = d.select(labelkeep='track', copy=False)
    d.average()
    return d



def fit_tipping(T_sys,SpillOver,pol,freqs):
    """Fit tipping curve.
        T_sys(el) = T_cmb + T_gal + T_atm*(1-exp(-tau_0/sin(el))) + T_spill(el) + T_rx
        We will fit the opacity and T_rx 
        T_cmb + T_gal is obtained from the T_sys.Tsky() function 
    """
#TODO Set this up to take in RA,dec not el to avoid scp problems
    returntext = [] # a list of Text to print to pdf
    T_atm = 1.12 * (273.15 + T_sys.surface_temperature) - 50.0 # ??
    # Create a function to give the spillover at any elevation at the observing frequency
    # Set up full tipping equation y = f(p, x):
    #   function input x = elevation in degrees
    #   parameter vector p = [T_rx, zenith opacity tau_0]
    #   function output y = T_sys in kelvin
    #   func = lambda p, x: p[0] + T_cmb + T_gal + T_spill_func(x) + T_atm * (1 - np.exp(-p[1] / np.sin(deg2rad(x))))
    #T_sky = np.average(T_sys.T_sky)# T_sys.Tsky(x)
    func = lambda p, x: p[0] +  T_sys.Tsky(x) + SpillOver.spill[pol](np.array([[x,],[freqs]])) + T_atm * (1 - np.exp(-p[1] / np.sin(np.radians(x))))
    # Initialise the fitter with the function and an initial guess of the parameter values
    tip = scape.fitting.NonLinearLeastSquaresFit(func, [70, 0.005])
    tip.fit(T_sys.elevation, T_sys.Tsys[pol])
    returntext.append('Fit results for %s polarisation:' % (pol,))
    returntext.append('T_ant %s = %.2f K' % (pol,tip.params[0],))
    returntext.append('Zenith opacity tau_0 %s= %.5f' % (pol,tip.params[1],))
    # Calculate atmospheric noise contribution at 10 degrees elevation for comparison with requirements
    T_atm_10 = T_atm * (1 - np.exp(-tip.params[1] / np.sin(deg2rad(10))))
    fit_func = []
    returntext.append('Atmospheric noise contribution at 10 degrees %s is: %.2f K' % (pol,T_atm_10,))
    for el in T_sys.elevation: fit_func.append(func(tip.params,el))
    chisq =chisq_pear(fit_func,T_sys.Tsys[pol])
    returntext.append('Chi square for %s is: %6f ' % (pol,chisq,))
    return {'params': tip.params,'fit':fit_func,'scatter': (T_sys.Tsys[pol]-fit_func),'chisq':chisq,'text':returntext}

def plot_data(Tsys,fit_HH,fit_VV):
    fig = plt.figure()
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
    plt.plot(Tsys.elevation, fit_HH['fit'], color='b' , label="HH" )
    plt.plot(Tsys.elevation, Tsys.Tsys['VV'], marker='^', color='r', label='VV measured', linewidth=0)
    plt.errorbar(Tsys.elevation, Tsys.Tsys['VV'],  Tsys.sigma_Tsys['VV'], ecolor='r', color='r', capsize=6, linewidth=0)
    plt.plot(Tsys.elevation, fit_VV['fit'], color='r', label="VV" )
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
    text = []
    text.append("HH $T_{rec}$ = %.2f K, $T_0$ = %.5f \n HH-$X^{2}$= %.3f" % (fit_HH['params'][0], fit_HH['params'][1], fit_HH['chisq']))
    text.append("VV $T_{rec}$ = %.2f K, $T_0$ = %.5f \n VV-$X^{2}$= %.3f" % (fit_VV['params'][0], fit_VV['params'][1], fit_VV['chisq']))
    return fig,text
    


def chisq_pear(fit,Tsys):
    fit = np.array(fit)
    return np.sum((Tsys-fit)**2/fit)


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a tipping curve plot in a pdf file.')
parser.add_option("-f", "--freq-chans", default='200,800',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option("-t", "--tip_models",
                  help="Name of directory containing spillover models")
parser.add_option( "--sky-map", default='~/comm/catalogues/TBGAL_CONVL.FITS',
                  help="Name of map of sky tempreture in fits format', default = '%default'")

(opts, args) = parser.parse_args()


if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')
        
h5 = katfile.open(args[0])
for ant in h5.ants:
    #Load the data file
    d = load_cal(args[0], "A%sA%s" % (ant.name[3:], ant.name[3:]), int(opts.freq_chans.split(',')[0]),int(opts.freq_chans.split(',')[1]))
    d.filename = [args[0]]
    nu = d.freqs[0]  #MHz Centre frequency of observation
    SpillOver = Spill_Temp(path=opts.tip_models)
    T_SysTemp = System_Temp(d,opts.sky_map,d.freqs[0])

    fit_H = fit_tipping(T_SysTemp,SpillOver,'HH',d.freqs)
    fit_V = fit_tipping(T_SysTemp,SpillOver,'VV',d.freqs)

    print ('Chi square for HH is: %6f ' % (fit_H['chisq'],))
    print ('Chi square for VV is: %6f ' % (fit_V['chisq'],))

    nice_filename =  args[0]+ '_' +d.antenna.name+'_tipping_curve'
    pp = PdfPages(nice_filename+'.pdf')
    T_SysTemp.sky_fig.savefig(pp,format='pdf')
    fig,text = plot_data(T_SysTemp,fit_H,fit_V)
    fig.savefig(pp,format='pdf')    
    fig = plt.figure(None,figsize = (10,16))
    plt.figtext(0.1,0.1,'\n'.join(text+fit_H['text']+fit_V['text']),fontsize=10)
    fig.savefig(pp,format='pdf')    
    plt.close('all')
    pp.close()




#
# Save option to be added
#


#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


#plt.xlabel(r'\textbf{time} (s)')
#plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
#plt.title(r"\TeX\ is Number "
#          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#          fontsize=16, color='gray')
