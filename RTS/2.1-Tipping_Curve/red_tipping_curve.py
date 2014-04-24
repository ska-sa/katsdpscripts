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
import katdal
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
    def __init__(self,filename=None):
        """ The class Spill_temp reads the spillover model from file and
        produces fitted functions for a frequency
        The class/__init__function takes in one parameter:
        filename : (default=none) This is the filename containing
               the spillover model ,this file has 3 cols:
               theta(Degrees, 0 at Zenith),tempreture (MHz),Frequency (MHz)
               if there are no files zero spilover is assumed.
               function save makes a file of the correct format
        returns :
               dict  spill with two elements 'HH' 'VV' that
               are intepolation functions that take in elevation & Frequency(MHz)
               and return tempreture in Kelven.
        """
#TODO Need to sort out better frequency interpolation & example
        try:
            spillover_H,spillover_V = np.loadtxt(filename).reshape(2,3,-1)
        except IOError:
            spillover_H = np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1200.,1200.,2000.,2000.]])
            spillover_V = np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1200.,1200.,2000.,2000.]])
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

    def save(filename=None,HH=np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1200.,1200.,2000.,2000.]]),VV=np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1200.,1200.,2000.,2000.]])):
        """ Save a Spillover model in the correct format
        filename : String filename
        HH,VV    : ndarray with shape (3,N )
                 : HH[0,:] is elevation degrees from zenith
                 : HH[1,:] is tempreture
                 : HH[2,:] is frequency
         the file is read using the command HH,VV = np.loadtxt('SpilloverModel.dat').reshape(2,3,-1)
        """
        if filename is None :
            print(help(self.save))
        else :
            with file(filename, 'w') as outfile:
                np.savetxt(outfile,HH)
                np.savetxt(outfile,VV)

class Rec_Temp:
    """Load Receiver models and interpolate to centre observing frequency."""
    def __init__(self,filename=''):
        """ The class Rec_temp reads the receiver model from file and
        produces fitted functions for a frequency
        The class/__init__function takes in one parameter:
        filename : (default='') This is the filename
               of the recever model
               these files have 2 cols:
                Frequency (MHz),tempreture (MHz),
               if there are no file 15k recever  is assumed.
        returns :
               dict  spill with two elements 'HH' 'VV' that
               are intepolation functions that take in Frequency(MHz)
               and return tempreture in Kelven.
        """
        try:
             receiver_h,receiver_v = np.loadtxt(filename, unpack=True)
        except IOError:
            receiver_h = np.array([[900.,2000],[15.,15.]])
            receiver_v = np.array([[900.,2000],[15.,15.]])
            warnings.warn('Warning: Failed to load Receiver models, setting models to 15 K ')
        #Assume  Provided models are a function of zenith angle & frequency
        T_H = fit.Spline1DFit(degree=1)
        T_V = fit.Spline1DFit(degree=1)
        T_H.fit(receiver_h[0],receiver_h[1])
        T_V.fit(receiver_v[0],receiver_v[1])
        self.rec = {}
        self.rec['HH'] = T_H # The HH and VV is a scape thing
        self.rec['VV'] = T_V

    def save(filename=None,HH=np.array([[900.,2000],[15.,15.]]),VV=np.array([[900.,2000],[15.,15.]])):
        """ Save a Recever model in the correct format
        filename : String filename
        HH,VV    : ndarray with shape (3,N )
                 : HH[0,:] is frequency
                 : HH[1,:] is tempreture
                 :
         the file is read using the command HH,VV = np.loadtxt('ReceverModel.dat').reshape(2,2,-1)
        """
        if filename is None :
            print(help(self.save))
        else :
            with file(filename, 'w') as outfile:
                np.savetxt(outfile,HH)
                np.savetxt(outfile,VV)


class System_Temp:
    """Extract tipping curve data points and surface temperature."""
    def __init__(self,d,path='TBGAL_CONVL.FITS',freqs=1822):#d, nu, pol
        """ First extract total power in each scan (both mean and standard deviation) """
        T_skytemp = Sky_temp(inputfile=path,nu=freqs)
        T_skytemp.set_freq(freqs)
        #print freqs
        T_sky =  T_skytemp.Tsky
        self.units = d.data_unit
        self.inputpath = path
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
        sort_ind  = elevation.argsort()
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

    def sky_fig(self):
        T_skytemp = Sky_temp(inputfile=self.inputpath,nu=self.freq)
        T_skytemp.set_freq(self.freq)
        return T_skytemp.plot_sky(self.ra,self.dec)


    def __iter__(self):
        return self

    def next(self):
        i = -1
        while True:
            i = i + 1
            if not self.ra[i]:raise StopIteration
            yield i,self.ra[i],self.dec[i],self.elevation[i]


###########################End Classes
def remove_rfi(d,width=3,sigma=5,axis=1):
    for i in range(len(d.scans)):
        d.scans[i].data = scape.stats.remove_spikes(d.scans[i].data,axis=axis,spike_width=width,outlier_sigma=sigma)
    return d


def load_cal(filename, baseline, freq_channel=None,channel_bw=10.0):
    """ Load the dataset into memory """
    d = scape.DataSet(filename, baseline=baseline)#, nd_models=nd_models
    if not freq_channel is None :
        d = d.select(freqkeep=freq_channel)
    print "Flagging RFI"
    #sd = remove_rfi(d,width=7,sigma=5)  # rfi flaging Needed ?
    print "Converting to Tempreture"
    d = d.convert_power_to_temperature()
    if not d is None:
        d = d.select(flagkeep='~nd_on')
        d = d.select(labelkeep='track', copy=False)
        d.average()
    return d



def fit_tipping(T_sys,SpillOver,pol,freqs,T_rx,fixopacity=False):
    """The 'tipping curve' is fitted using the expression below, with the free parameters of $T_{ant}$ and $\tau_{0}$
        the Antenna tempreture and the atmospheric opacity. All the varables are also functions of frequency .
        $T_{sys}(el) = T_{cmb}(ra,dec) + T_{gal}(ra,dec) + T_{atm}*(1-\exp(\frac{-\ta   u_{0}}{\sin(el)})) + T_spill(el) + T_{ant} + T_{rx}$
        We will fit the opacity and $T_{ant}$.s
        T_cmb + T_gal is obtained from the T_sys.Tsky() function
        if fixopacity is set to true then $\tau_{0}$ is set to 0.01078 (Van Zee et al.,1997) this means that $T_{ant}$ becomes
        The excess tempreture since the other components are known. When fixopacity is not True then it is fitted and T_ant
        is assumed to be constant with elevation
    """
#TODO Set this up to take in RA,dec not el to avoid scp problems
    T_atm = 1.12 * (273.15 + T_sys.surface_temperature) - 50.0 # This is some equation
    returntext = []
    if not fixopacity:
        # a list of Text to print to pdf
        # Create a function to give the spillover at any elevation at the observing frequency
        # Set up full tipping equation y = f(p, x):
        #   function input x = elevation in degrees
        #   parameter vector p = [T_rx, zenith opacity tau_0]
        #   function output y = T_sys in kelvin
        #   func = lambda p, x: p[0] + T_cmb + T_gal  + T_spill_func(x) + T_atm * (1 - np.exp(-p[1] / np.sin(deg2rad(x))))
        #T_sky = np.average(T_sys.T_sky)# T_sys.Tsky(x)
        func = lambda p, x: p[0] + T_rx.rec[pol](freqs)+  T_sys.Tsky(x) + SpillOver.spill[pol](np.array([[x,],[freqs]])) + T_atm * (1 - np.exp(-p[1] / np.sin(np.radians(x))))
        # Initialise the fitter with the function and an initial guess of the parameter values
        tip = scape.fitting.NonLinearLeastSquaresFit(func, [30, 0.01])
        tip.fit(T_sys.elevation, T_sys.Tsys[pol])
        returntext.append('Fit results for %s polarisation at %.1f Mhz:' % (pol,np.mean(freqs)))
        returntext.append('$T_{ant}$ %s = %.2f %s  at %.1f Mhz' % (pol,tip.params[0],T_sys.units,np.mean(freqs)))
        returntext.append('Zenith opacity $tau_{0}$ %s= %.5f  at %.1f Mhz' % (pol,tip.params[1],np.mean(freqs)))
        fit_func = []
        for el in T_sys.elevation: fit_func.append(func(tip.params,el))
        chisq =chisq_pear(fit_func,T_sys.Tsys[pol])
        returntext.append('$\chi^2$ for %s is: %6f ' % (pol,chisq,))
        # Calculate atmosphesric noise contribution at 10 degrees elevation for comparison with requirements
        #T_atm_10 = T_atm * (1 - np.exp(-tip.params[1] / np.sin(deg2rad(10))))#Atmospheric noise contribution at 10 degrees
    else:
        tau = 0.01078
        tip = scape.fitting.NonLinearLeastSquaresFit(None, [0, 0.00]) # nonsense Vars
        func = lambda x, tsys: tsys - (T_rx.rec[pol](freqs)+  T_sys.Tsky(x) + SpillOver.spill[pol](np.array([[x,],[freqs]])) + T_atm * (1 - np.exp(-tau / np.sin(np.radians(x)))))
        fit_func = []
        returntext.append('Not fitting Opacity assuming a value if %f , $T_{ant}$ is the residual of of model data. ' % (tau,))
        for el,t_sys in zip(T_sys.elevation, T_sys.Tsys[pol]): fit_func.append(func(el,t_sys))
        chisq =0.0# nonsense Vars
    return {'params': tip.params,'fit':fit_func,'scatter': (T_sys.Tsys[pol]-fit_func),'chisq':chisq,'text':returntext}

def plot_data_el(Tsys,Tant,title='',units='K',line=42):
    fig = plt.figure()
    elevation = Tsys[:,2]
    line1,=plt.plot(elevation, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(elevation, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(elevation, Tant[:,0], color='b'  )
    line3,=plt.plot(elevation, Tsys[:,1], marker='^', color='r', linewidth=0)
    plt.errorbar(elevation, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(elevation, Tant[:,1], color='r')
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}$ HH','$T_{ant}$ HH', '$T_{sys}$ VV','$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Elevation (degrees)')
    plt.ylim(np.min((Tsys[:,0:2].min(),Tant[:,0:2].min())),np.max((np.percentile(Tsys[:,0:2],90),np.percentile(Tant[:,0:2],90),line*1.1)))
    plt.hlines(line, elevation.min(), elevation.max(), colors='k')
    plt.grid()
    if units == 'K':
        plt.ylabel('Tempreture (K)')
    else:
        plt.ylabel('Raw power (counts)')
        plt.legend()
    return fig

def plot_data_freq(frequency,Tsys,Tant,title=''):
    fig = plt.figure()
    line1,=plt.plot(frequency, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(frequency, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(frequency, Tant[:,0], color='b'  )
    line3,=plt.plot(frequency, Tsys[:,1], marker='^', color='r',  linewidth=0)
    plt.errorbar(frequency, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(frequency, Tant[:,1], color='r')
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}$ HH','$T_{ant}$ HH', '$T_{sys}$ VV','$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Frequency (MHz)')
    plt.ylim(np.min((Tsys[:,0:2].min(),Tant[:,0:2].min())),np.max((np.percentile(Tsys[:,0:2],90),np.percentile(Tant[:,0:2],90),46*1.1)))
    if np.min(frequency) <= 1420 :
        plt.hlines(42, np.min((frequency.min(),1420)), 1420, colors='k')
    if np.max(frequency) >=1420 :
        plt.hlines(46, np.max((1420,frequency.min())), np.max((frequency.max(),1420)), colors='k')
    plt.grid()
    if units == 'K':
        plt.ylabel('Tempreture (K)')
    else:
        plt.ylabel('Raw power (counts)')
    return fig


def chisq_pear(fit,Tsys):
    fit = np.array(fit)
    return np.sum((Tsys-fit)**2/fit)


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a tipping curve plot in a pdf file.')
parser.add_option("-f", "--freq-chans", default=None,
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default= %default)")
parser.add_option("-r", "--select-freq", default='900,1420,1790,1840',
                  help="Range of averaged frequency channels to plot (comma delimated specified in MHz , default= %default)")
parser.add_option("-e", "--select-el", default='90,15,45',
                  help="Range of elevation scans to plot (comma delimated specified in Degrees abouve the Horizon , default= %default)")
parser.add_option("-b", "--freq-bw", default=10.0,
                  help="Bandwidth of frequency channels to average in MHz (, default= %default MHz)")
parser.add_option("-s", "--spill-over-models",default='',
                  help="Name of file containing spillover models")
parser.add_option( "--receiver-models",default='',
                  help="Name of file containing receiver models")
parser.add_option( "--fix-opacity",default=True,
                  help="This option has not been completed, Do not let opacity be a free parameter in the fit , this changes the fitting in to just a model subtraction and T_ant is the error")
parser.add_option( "--sky-map", default='TBGAL_CONVL.FITS',
                  help="Name of map of sky tempreture in fits format', default = '%default'")

(opts, args) = parser.parse_args()


if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()

select_freq= np.array(opts.select_freq.split(','),dtype=float)
select_el = np.array(opts.select_el.split(','),dtype=float)
h5 = katdal.open(args[0])
h5.select(ants='ant3',scans='track')
if not opts.freq_chans is None: h5.select(channels=slice(opts.freq_chans.split(',')[0],opts.freq_chans.split(',')[1]))
for ant in h5.ants:
    #Load the data file
    first = True
    #freq loop
    nice_filename =  args[0].split('/')[-1]+ '_' +ant.name+'_tipping_curve'
    pp =PdfPages(nice_filename+'.pdf')
    #T_SysTemp = System_Temp(d,opts.sky_map,h5.channel_freqs.mean()/1e6)
    #T_SysTemp.sky_fig.savefig(pp,format='pdf')
    channel_bw = opts.freq_bw
    num_channels = np.int(channel_bw/(h5.channel_width/1e6)) #number of channels per band
    chunks=[h5.channels[x:x+num_channels] for x in xrange(0, len(h5.channels), num_channels)]
    freq_list = np.zeros((len(chunks)))
    for j,chunk in enumerate(chunks):freq_list[j] = h5.channel_freqs[chunk].mean()/1e6
    tsys = np.zeros((len(h5.scan_indices),len(chunks),5 ))#*np.NaN
    tant = np.zeros((len(h5.scan_indices),len(chunks),5 ))#*np.NaN
    print "Selecting channel data to form %f MHz Channels"%(channel_bw)
    for i,chunk in enumerate(chunks):
        d = load_cal(args[0], "A%sA%s" % (ant.name[3:], ant.name[3:]), chunk)
        if not d is None:
            d.filename = [args[0]]
            nu = d.freqs  #MHz Centre frequency of observation
            SpillOver = Spill_Temp(filename=opts.spill_over_models)
            recever = Rec_Temp(filename=opts.receiver_models)
            T_SysTemp = System_Temp(d,opts.sky_map,d.freqs[0])
            units = T_SysTemp.units+''
            fit_H = fit_tipping(T_SysTemp,SpillOver,'HH',d.freqs,recever,fixopacity=opts.fix_opacity)
            fit_V = fit_tipping(T_SysTemp,SpillOver,'VV',d.freqs,recever,fixopacity=opts.fix_opacity)
            #print ('Chi square for HH  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_H['chisq'],))
            #print ('Chi square for VV  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_V['chisq'],))
            length = len(T_SysTemp.elevation)
            tsys[0:length,i,0] = T_SysTemp.Tsys['HH']
            tsys[0:length,i,1] = T_SysTemp.Tsys['VV']
            tsys[0:length,i,2] = T_SysTemp.elevation
            tsys[0:length,i,3] = T_SysTemp.sigma_Tsys['HH']
            tsys[0:length,i,4] = T_SysTemp.sigma_Tsys['VV']
            tant[0:length,i,0] = fit_H['fit']
            tant[0:length,i,1] = fit_V['fit']
            tant[0:length,i,2] = T_SysTemp.elevation

            #break

            #fig,text = plot_data(T_SysTemp,fit_H,fit_V)
            if first :
                fig = T_SysTemp.sky_fig()
                fig.savefig(pp,format='pdf')
                first = False
                plt.close()
    for freq in select_freq :
        title = ""
        if np.abs(freq_list-freq).min() < opts.freq_bw*1.1 :
            i = (np.abs(freq_list-freq)).argmin()
            lineval = 42
            if freq > 1420 : lineval = 46
            fig = plot_data_el(tsys[0:length,i,:],tant[0:length,i,:],title=r"$T_{sys}$ and $T_{ant}$ at %.1f MHz"%(freq),units=units,line=lineval)
            fig.savefig(pp,format='pdf')
    for el in select_el :
        title = ""
        i = (np.abs(tsys[0:length,:,2].max(axis=1)-el)).argmin()
        fig = plot_data_freq(freq_list,tsys[i,:,:],tant[i,:,:],title=r"$T_{sys}$ and $T_{ant}$ at %.1f Degrees elevation"%(np.abs(tsys[0:length,:,2].max(axis=1))))
        fig.savefig(pp,format='pdf')

    fig = plt.figure(None,figsize = (8,8))
    text =r"""The 'tipping curve' is calculated according to the expression below,
with the the parameters of $T_{ant}$ and $\tau_{0}$,
the Antenna tempreture and the atmospheric opacity respectivly.
All the varables are also functions of frequency.

$T_{sys}(el) = T_{cmb}(ra,dec) + T_{gal}(ra,dec) + T_{atm}*(1-\exp(\frac{-\tau_{0}}{\sin(el)})) + T_{spill}(el) + T_{ant} + T_{rx}$

$T_{sys}(el)$ is determined from the noise diode calibration
so it is $\frac{T_{sys}(el)}{\eta_{illum}}$.
We assume the opacity and $T_{ant}$ is the residual after
the tipping curve function is calculated. T_cmb + T_gal is
obtained from the Sky model. $\tau_{0}$, the opacity,
is set to 0.01078 (Van Zee et al.,1997). $T_{ant}$ is the excess
tempreture since the other components are known."""

    plt.figtext(0.1,0.1,text,fontsize=10)
    fig.savefig(pp,format='pdf')
    pp.close()
    plt.close('all')





#
# Save option to be added
#

#d = load_cal('1386872829.h5', "A3A3", None)

#for scan in d.scans:plt.plot(d.freqs,scan.data[:,:,0].mean(axis=0))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


#plt.xlabel(r'\textbf{time} (s)')
#plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
#plt.title(r"\TeX\ is Number "
#          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#          fontsize=16, color='gray')
