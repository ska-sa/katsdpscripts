#!/usr/bin/python
# Reduces tipping curve data and fits a tipping model to the data
# Needs: noise diode models and tipping spillover model

import optparse
import numpy as np
import matplotlib.pyplot as plt

import pickle
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import katdal
import scape
import scikits.fitting as fit
import gsm
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from katsdpscripts import git_info
from matplotlib.offsetbox import AnchoredText


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

class Sky_temp:
    import gsm
    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    """
       T_sky = T_cont + T_cmb  from the global sky model
       Read in  file, and provide a method of passing back the Tsky temp a at a position
    """

    def __init__(self,nu=1828.0,path="/var/kat/archive/data/models/gsm",diameter=13.5,smooth=True):
        """ Load The Tsky data from an inputfile in FITS format and scale to frequency
        This takes in 1 parameter:
        nu (MHz) center frequency
        """
        if smooth :
            self.freq_map = hp.sphtfunc.smoothing(gsm.get_freq(nu,path),fwhm=(1.17*(3e8/(nu*1e6))/diameter ) )
        else :
            self.freq_map = gsm.get_freq(nu,path)
        self.freq_map += 2.725  # CMB tempreture not included in de Oliveira-Costa's GSM
        self.nu = nu
        self.smooth=smooth

    def Tsky(self,ra,dec):
        """given RA/Dec in Degrees  return the value of the spot
        assuming the healpix map is in Galatic coords
        """
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        l = c.galactic.l.radian
        b = c.galactic.b.radian
        nside = hp.npix2nside(self.freq_map.shape[0])
        ipix = hp.ang2pix(nside, np.pi/2.0 - b , l % (np.pi*2),nest=False)
        return self.freq_map[ipix]


    def plot_sky(self,ra=None,dec=None,norm = 'log',unit='Kelvin',heapix_array=None):
        """ plot_sky plots the sky tempreture and overlays pointing centers as red dots
        The sky tempreture is the data that was loaded when the class was iniitated.
        plot_sky takes in 3 optional parameters:
                ra,dec  are list/1D-array like values of right assension and declanation
        returns matplotlib figure object that the plot is assosated with.
        """
        #self.freq_map
        fig = plt.figure(figsize=(16,9))
        hp.cartview(self.freq_map,norm = norm,unit=unit,fig=fig.number)
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        l = np.degrees(angle_wrap(-c.galactic.l.radian % (np.pi*2)) )
        b = np.degrees(c.galactic.b.radian)
        plt.plot(l,b,'ro')
        #hp.graticule()
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
            datafile =np.loadtxt(filename)#  First line is Frequencys, First Col is deg from zenith after that cols are in alternating pol
            elevation = datafile[1:,0]
            numfreqs = (datafile.shape[1]-1)//2
            freqs= datafile[0,1::2]
            elevation_list = np.array(())
            freq_list = np.array(())
            data_list = np.array(())
            elevation_list = np.r_[elevation_list,elevation]
            freq_list = np.r_[freq_list,np.ones_like(elevation)*(freqs.min()-100)] ## Extend the lower limit to avoid nans
            data_list = np.r_[data_list,datafile[1:,1+0*2]]
            for x in range(numfreqs):
                elevation_list = np.r_[elevation_list,elevation]
                freq_list = np.r_[freq_list,np.ones_like(elevation)*freqs[x]]
                data_list = np.r_[data_list,datafile[1:,1+x*2]]
            elevation_list = np.r_[elevation_list,elevation]
            freq_list = np.r_[freq_list,np.ones_like(elevation)*(freqs.max()+100)] ## Extend the upper limit to avoid nans
            data_list = np.r_[data_list,datafile[1:,1+(numfreqs-1)*2]]

            T_H = fit.Delaunay2DScatterFit()
            T_H.fit((90.-elevation_list,freq_list),data_list)
            elevation_list = np.array(())
            freq_list = np.array(())
            data_list = np.array(())
            elevation_list = np.r_[elevation_list,elevation]
            freq_list = np.r_[freq_list,np.ones_like(elevation)*(freqs.min()-100)]  ## Extend the lower limit to avoid nans
            data_list = np.r_[data_list,datafile[1:,1+0*2+1]]
            for x in range(numfreqs):
                elevation_list = np.r_[elevation_list,elevation]
                freq_list = np.r_[freq_list,np.ones_like(elevation)*freqs[x]]
                data_list = np.r_[data_list,datafile[1:,1+x*2+1]] 
            elevation_list = np.r_[elevation_list,elevation]
            freq_list = np.r_[freq_list,np.ones_like(elevation)*(freqs.max()+100)] ## Extend the upper limit to avoid nans
            data_list = np.r_[data_list,datafile[1:,1+(numfreqs-1)*2+1]]
            T_V = fit.Delaunay2DScatterFit()
            T_V.fit((90.-elevation_list,freq_list),data_list)
            self.spill = {}
            self.spill['HH'] = T_H # The HH and VV is a scape thing
            self.spill['VV'] = T_V
            #print self.spill['HH']((90.-elevation_list,freq_list))

        except IOError:
            spillover_H = np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1.,1.,2000.,2000.]])
            spillover_V = np.array([[0.,90.,0.,90.],[0.,0.,0.,0.],[1.,1.,2000.,2000.]])
            spillover_H[0]= 90-spillover_H[0]
            spillover_V[0]= 90-spillover_V[0]
            T_H = fit.Delaunay2DScatterFit()
            T_V = fit.Delaunay2DScatterFit()
            T_H.fit(spillover_H[[0,2],:],spillover_H[1,:])
            T_V.fit(spillover_V[[0,2],:],spillover_V[1,:])
            self.spill = {}
            self.spill['HH'] = T_H # The HH and VV is a scape thing
            self.spill['VV'] = T_V
            warnings.warn('Warning: Failed to load Spillover models, setting models to zeros')
            print('Warning: Failed to load Spillover models, setting models to zeros')
        # the models are in a format of theta=0  == el=90


class aperture_efficiency_models:
    """Load aperture_efficiency_models and interpolate to centre observing frequency."""
    def __init__(self,filenameH='',filenameV='',):
        """ The class aperture_efficiency reads the aperture_efficiency
        model from file and
        produces fitted functions for a frequency
        The class/__init__function takes in one parameter:
        filename : (default='') This is the filename
               of the recever model
               these files have 2 cols:
                Frequency (MHz),aperture_efficiency  (fraction),
               if there are no file 15k recever  is assumed.
        returns :
               dict  spill with two elements 'HH' 'VV' that
               are intepolation functions that take in Frequency(MHz)
               and return fraction.
        """
        try:
            aperture_eff_h = np.loadtxt(filenameH,comments='#')# Change units to fraction
            a800 = np.zeros((aperture_eff_h.shape[0]+2,2))
            a800[0,:] = [aperture_eff_h[0,0]-100,aperture_eff_h[0,1]]# Extend the model by 100 MHz
            a800[1:-1,:] = aperture_eff_h
            a800[-1,:] = [aperture_eff_h[-1,0]+100,aperture_eff_h[-1,1]]# Extend the model by 100 MHz
            aperture_eff_h = a800

            aperture_eff_v = np.loadtxt(filenameV,comments='#')# Change units to fraction
            a800 = np.zeros((aperture_eff_v.shape[0]+2,2))
            a800[0,:] = [aperture_eff_v[0,0]-100,aperture_eff_v[0,1]]# Extend the model by 100 MHz
            a800[1:-1,:] = aperture_eff_v
            a800[-1,:] = [aperture_eff_v[-1,0]+100,aperture_eff_v[-1,1]]# Extend the model by 100 MHz
            aperture_eff_v = a800
        except IOError:
            aperture_eff_h = np.array([[1.,75.],[2000.,75.]])
            aperture_eff_v = np.array([[1.,75.],[2000.,75.]])
            warnings.warn('Warning: Failed to load aperture_efficiency models, setting models to 0.75 ')
            print('Warning: Failed to load aperture_efficiency models, setting models to 0.75 ')
        #Assume  Provided models are a function of zenith angle & frequency
        T_H = fit.PiecewisePolynomial1DFit()
        T_V = fit.PiecewisePolynomial1DFit()
        T_H.fit(aperture_eff_h[:,0],aperture_eff_h[:,1]/100.)
        T_V.fit(aperture_eff_v[:,0],aperture_eff_v[:,1]/100.)
        self.eff = {}
        self.eff['HH'] = T_H # The HH and VV is a scape thing
        self.eff['VV'] = T_V

class Rec_Temp:
    """Load Receiver models and interpolate to centre observing frequency."""
    def __init__(self,filenameH='',filenameV=''):
        """ The class Rec_temp reads the receiver model from file and
        produces fitted functions for a frequency
        The class/__init__function takes in one parameter:
        filenameH : (default='') This is the filename
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
            receiver_h = (np.loadtxt(filenameH,comments='%',delimiter=',')[:,[0,2] ]/(1e6,1.)).T # Change units to MHz # discard the gain col
            a800 = np.zeros((2,np.shape(receiver_h)[-1]+1))
            a800[:,0] = [receiver_h[0,0]-100,receiver_h[1,0]]  # Extend the model by 100 MHz
            a800[:,1:] = receiver_h
            receiver_h = a800
            receiver_v = (np.loadtxt(filenameV,comments='%',delimiter=',')[:,[0,2] ]/(1e6,1.)).T # Change units to MHz  # discard the gain col
            a800 = np.zeros((2,np.shape(receiver_v)[-1]+1))
            a800[:,0] = [receiver_v[0,0]-100,receiver_v[1,0]] # Extend the model by 100 MHz
            a800[:,1:] = receiver_v
            receiver_v = a800
        except IOError:
            receiver_h = np.array([[1.,20.],[2000.,20.]])
            receiver_v = np.array([[1.,20.],[2000.,20.]])
            warnings.warn('Warning: Failed to load Receiver models, setting models to 20 K ')
            print('Warning: Failed to load Receiver models, setting models to 20 K ')
        #Assume  Provided models are a function of zenith angle & frequency
        T_H = fit.PiecewisePolynomial1DFit()
        T_V = fit.PiecewisePolynomial1DFit()
        T_H.fit(receiver_h[0],receiver_h[1])
        T_V.fit(receiver_v[0],receiver_v[1])
        self.rec = {}
        self.rec['HH'] = T_H # The HH and VV is a scape thing
        self.rec['VV'] = T_V

class System_Temp:
    """Extract tipping curve data points and surface temperature."""
    def __init__(self,d,freqs=1822,freq_index=0,elevation=None,ra=None,dec=None ,surface_temperature=23.0,air_relative_humidity=0.23):#d, nu, pol
        """ First extract total power in each scan (both mean and standard deviation) """
        T_skytemp = Sky_temp(nu=freqs)
        T_sky =  T_skytemp.Tsky
        self.units = d.data_unit

        self.name = d.antenna.name
        self.filename = d.filename
        self.elevation =  {}
        self.Tsys = {}
        self.sigma_Tsys = {}
        self.Tsys_sky = {}
        self.T_sky = []
        self.height = d.antenna.position_wgs84[2]
        self.pressure =  np.mean([line[1] for line in d.enviro['pressure'] ])
        self.air_relative_humidity = air_relative_humidity
        # Sort data in the order of ascending elevation
        valid_el = (elevation >= 10)
        self.elevation =  elevation[valid_el]
        self.ra = ra[valid_el]
        self.dec = dec[valid_el]
        self.surface_temperature = surface_temperature# Extract surface temperature from weather data
        self.freq = freqs  #MHz Centre frequency of observation
        for pol in ['HH','VV']:
            power_stats = [scape.stats.mu_sigma(s.pol(pol)[:,freq_index]) for s in d.scans]
            tipping_mu, tipping_sigma = np.array([s[0] for s in power_stats]), np.array([s[1] for s in power_stats])
            tipping_mu, tipping_sigma = tipping_mu[sort_ind], tipping_sigma[sort_ind]
            self.Tsys[pol] = tipping_mu[valid_el]
            self.sigma_Tsys[pol] = tipping_sigma[valid_el]
            self.Tsys_sky[pol] = []
            self.T_sky = []
            for val_el,ra,dec,el in zip(sort_ind,self.ra,self.dec,self.elevation):
                self.T_sky.append( T_sky(ra,dec))
                self.Tsys_sky[pol].append(tipping_mu[val_el]-T_sky(ra,dec))
        TmpSky = fit.PiecewisePolynomial1DFit()
        TmpSky.fit(self.elevation, self.T_sky)
        self.Tsky = TmpSky

    def sky_fig(self,freq=1328):
        T_skytemp = Sky_temp(nu=freq)
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

def load_cal(filename, baseline, nd_models, freq_channel=None,channel_bw=10.0,channel_mask='',n_chan = 4096,channel_range=None,band_input=None):
    """ Load the dataset into memory """
    print('Loading noise diode models')

    try:
        d = scape.DataSet(filename, baseline=baseline, nd_models=nd_models,band=band_input)
    except IOError:
        nd = scape.gaincal.NoiseDiodeModel(freq=[1,2000],temp=[20,20])
        warnings.warn('Warning: Failed to load/find Noise Diode Models, setting models to 20K ')
        print('Warning: Failed to load/find Noise Diode Models, setting models to 20K ')
        d = scape.DataSet(filename, baseline=baseline,  nd_h_model = nd, nd_v_model=nd ,band=band_input)


    if not channel_range is None :
        start_freq_channel = int(channel_range.split(',')[0])
        end_freq_channel = int(channel_range.split(',')[1])
        edge = np.tile(True, n_chan)
        edge[slice(start_freq_channel, end_freq_channel)] = False
    else :
        edge = np.tile(False, n_chan)
    #load static flags if pickle file is given
    if len(channel_mask)>0:
        pickle_file = open(channel_mask)
        rfi_static_flags = pickle.load(pickle_file)
        pickle_file.close()
    else:
        rfi_static_flags = np.tile(False, n_chan)

    static_flags = np.logical_or(edge,rfi_static_flags)

    #d = d.select(freqkeep=~static_flags)
    freq_channel_flagged = []
    for band in freq_channel:
        tmp_band = []
        for channel in band :
            if not static_flags[channel] : # if not flagged
                tmp_band.append(channel)
        #if len(tmp_band) > 0 :
        freq_channel_flagged.append(tmp_band)

    #if not freq_channel is None :
    #    d = d.select(freqkeep=freq_channel)
    #print "Flagging RFI"
    #sd = remove_rfi(d,width=7,sigma=5)  # rfi flaging Needed ?
    print "Converting to Tempreture"
    print "Plotting the number of channels in each band of the list of lists freq_channel_flagged will be usefull "
    d = d.convert_power_to_temperature(freq_width=0.0)
    if not d is None:
        d = d.select(flagkeep='~nd_on')
        d = d.select(labelkeep='track', copy=False)
        d.average(channels_per_band=freq_channel_flagged)
    return d



def chisq_pear(fit,Tsys):
    fit = np.array(fit)
    return np.sum((Tsys-fit)**2/fit)

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
        #print T_sys.surface_temperature,T_sys.air_relative_humidity, T_sys.pressure, T_sys.height, freqs
        tau = calc_atmospheric_opacity(T_sys.surface_temperature,T_sys.air_relative_humidity, T_sys.pressure, T_sys.height/1000., freqs/1000.)
        # Height in meters above sea level, frequency in GHz.
    else:
        tau = 0.01078
    print("atmospheric_opacity = %f  at  %f MHz"%(tau,freqs))
    tip = fit.NonLinearLeastSquaresFit(None, [0, 0.00]) # nonsense Vars
    def know_quant(x):
        rx = T_rx.rec[pol](freqs)
        sky = T_sys.Tsky(x)
        spill = SpillOver.spill[pol](np.array([[x,],[freqs]]))
        atm = T_atm * (1 - np.exp(-tau / np.sin(np.radians(x))))
        #print "Rec %3.1f + Sky %3.1f + Spill %3.1f + Atm %3.1f = %3.1f" % (rx ,sky , spill , atm,rx+sky+spill+atm)
        return rx + sky + spill + atm

    func = know_quant
    fit_func = []
    returntext.append('Not fitting Opacity assuming a value if %f , $T_{ant}$ is the residual of of model data. ' % (tau,))
    for el,t_sys in zip(T_sys.elevation, T_sys.Tsys[pol]):
        fit_func.append(t_sys - func(el))
        #print "T_sys %3.1f - T_other %3.1f " %(t_sys,func(el))
    chisq =0.0# nonsense Vars
    return {'params': tip.params,'fit':fit_func,'scatter': (T_sys.Tsys[pol]-fit_func),'chisq':chisq,'text':returntext}

def plot_data_el(Tsys,Tant,title='',units='K',line=42,aperture_efficiency=None,frequency=1420):
    fig = plt.figure(figsize=(16,9))
    elevation = Tsys[:,2]
    line1,=plt.plot(elevation, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(elevation, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(elevation, Tant[:,0], color='b'  )
    line3,=plt.plot(elevation, Tsys[:,1], marker='^', color='r', linewidth=0)
    plt.errorbar(elevation, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(elevation, Tant[:,1], color='r')
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}/\eta_{ap}$ HH','$T_{ant}$ HH', '$T_{sys}/\eta_{ap}$ VV','$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Elevation (degrees)')
    lim_min = r_lim([np.percentile(Tsys[:,0:2],10),np.percentile(Tant[:,0:2],10),-5.])
    if line is not None:
        linev = line
    else:
        linev= 0
    if line is not None:
        plt.hlines(line, elevation.min(), elevation.max(), colors='k')
    if aperture_efficiency is not None:
        recLim_apEffH = receptor_band_limit(frequency,elevation)/aperture_efficiency.eff['HH'](frequency)
        recLim_apEffV = receptor_band_limit(frequency,elevation)/aperture_efficiency.eff['VV'](frequency)
        plt.plot(elevation,recLim_apEffH, lw=1.1,c='g',linestyle='-')
        plt.plot(elevation,recLim_apEffV, lw=1.1,c='g',linestyle='-')
        #print recLim_apEffH.shape,frequency
        for error_margin in [0.9,1.1]:
            plt.plot(elevation,recLim_apEffH*error_margin, lw=1.1,c='g',linestyle='--')
            plt.plot(elevation,recLim_apEffV*error_margin, lw=1.1,c='g',linestyle='--')
    lim_max = r_lim([np.percentile(Tsys[:,0:2],90),np.percentile(Tant[:,0:2],90)*1.1,np.max(recLim_apEffH)*1.2,linev*1.1],np.max)
    plt.ylim(lim_min,lim_max)
    plt.grid()
    plt.ylabel('$T_{sys}/\eta_{ap}$  (K)')
    return fig

def r_lim(dataf,func=np.min):
    """ Returns the func of the data , not used on nans"""
    dataf = np.array(dataf)
    index = ~np.isnan(dataf)
    return func(dataf[index,...])


def receptor_band_limit(frequency,elevation):
    
    if (frequency.min() < 800) :
        return receptor_UHFband_limit(frequency,elevation)
    else :
        return receptor_Lband_limit(frequency,elevation)

def receptor_Lband_limit(frequency,elevation): # APH added elevation
    """275-410 m^2/K at Receivers CDR"""
    #print frequency.shape,elevation.shape
    #print frequency,elevation
    return_array = np.zeros_like(frequency,dtype=np.float) # APH changed division below to "/float()"
    return_array[np.array(frequency < 1280)] = np.array(12 + 6+(5.5-6)/float(1280-900)*(frequency-900))[np.array(frequency < 1280)]
    return_array[np.array(~(frequency < 1280))] = np.array(12 + 5.5+(4-5.5)/float(1670-1280)*(frequency-1280))[np.array(~(frequency < 1280))]
    # APH added the code below to adjust for atmosphere vs. elevation
    Tatm = lambda f_MHz, el: 275*(1-np.exp(-(0.005+0.075*(f_MHz/22230.)**4)/np.sin(el*np.pi/180))) # Approximate relation appropriate for spec limit
    DT_elevation = Tatm(frequency,elevation)-Tatm(frequency,90)
    return return_array+DT_elevation

def receptor_UHFband_limit(frequency,elevation): # APH added elevation
    #print frequency.shape,elevation.shape
    #print frequency,elevation
    return_array = np.zeros_like(frequency,dtype=np.float) # APH changed division below to "/float()"
    return_array[np.array(frequency < 900)] = np.array(8 + (12-8)/float(1015-580)*(frequency-580) + 8+(7-8)/float(900-580)*(frequency-580))[np.array(frequency < 900)]
    return_array[np.array(~(frequency < 900))] = np.array (8 + (12-8)/float(1015-580)*(frequency-580) + 7+(4-7)/float(1015-900)*(frequency-900))[np.array(~(frequency < 900))]
    # APH added the code below to adjust for atmosphere vs. elevation
    Tatm = lambda f_MHz, el: 275*(1-np.exp(-(0.005+0.075*(f_MHz/22230.)**4)/np.sin(el*np.pi/180))) # Approximate relation appropriate for spec limit
    DT_elevation = Tatm(frequency,elevation)-Tatm(frequency,90)
    return return_array+DT_elevation
    

def plot_data_freq(frequency,Tsys,Tant,title='',aperture_efficiency=None,band='L'):
    fig = plt.figure(figsize=(16,9))
    line1,=plt.plot(frequency, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(frequency, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(frequency, Tant[:,0], color='b'  )
    line3,=plt.plot(frequency, Tsys[:,1], marker='^', color='r',  linewidth=0)
    plt.errorbar(frequency, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(frequency, Tant[:,1], color='r')
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}/\eta_{ap}$ HH','$T_{ant}$ HH', '$T_{sys}/\eta_{ap}$ VV','$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Frequency (MHz)')
    if aperture_efficiency is not None:
        #print Tsys[:,2]
        recLim_apEffH = receptor_band_limit(frequency,Tsys[:,2])/aperture_efficiency.eff['HH'](frequency)
        recLim_apEffV = receptor_band_limit(frequency,Tsys[:,2])/aperture_efficiency.eff['VV'](frequency)
        plt.plot(frequency,recLim_apEffH,lw=1.1,color='limegreen',linestyle='-')
        plt.plot(frequency,recLim_apEffV,lw=1.1,color='limegreen',linestyle='-')
        for error_margin in [0.9,1.1]:
            plt.plot(frequency,recLim_apEffH*error_margin, lw=1.1,color='g',linestyle='--')
            plt.plot(frequency,recLim_apEffV*error_margin, lw=1.1,color='g',linestyle='--')

    low_lim = (r_lim(Tsys[:,0:2]),r_lim(Tant[:,0:2]) )
    low_lim = np.min(low_lim)
    low_lim = -5. # np.max((low_lim , -5.))
    def tmp(x):
        return np.percentile(x,80)
    high_lim = (r_lim(Tsys[:,0:2],tmp),r_lim(Tant[:,0:2],tmp))
    high_lim = np.max(high_lim)
    high_lim = np.max((high_lim , 46*1.3))
    plt.ylim(low_lim,high_lim)
    if band=='L':
        plt.vlines(900,low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
        plt.vlines(1680,low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
        if np.min(frequency) <= 1420 :
            plt.hlines(42, np.min((frequency.min(),1420)), 1420, colors='k')
        if np.max(frequency) >=1420 :
            plt.hlines(46, np.max((1420,frequency.min())), np.max((frequency.max(),1420)), colors='k')
    if band=='U':
        plt.vlines(580,low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
        plt.vlines(1050,low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
    plt.grid()
    plt.ylabel('$T_{sys}/\eta_{ap}$  (K)')
    return fig


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a tipping curve plot in a pdf file.')
parser.add_option("-f", "--freq-chans", default=None,
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default= %default)")
parser.add_option("-r", "--select-freq", default='600,700,800,900,1440,1670,1840,2000,2250,2500,2750,3000,3250,3500,3750,4000',
                  help="Range of averaged frequency channels to plot (comma delimited specified in MHz , default= %default)")
parser.add_option("-e", "--select-el", default='90,15,45',
                  help="Range of elevation scans to plot (comma delimated specified in Degrees abouve the Horizon , default= %default)")
parser.add_option("-b", "--freq-bw", default=10.0,type="float",
                  help="Bandwidth of frequency channels to average in MHz (, default= %default MHz)")
parser.add_option("-s", "--spill-over-models",default='/var/kat/katconfig/user/spillover-models/mkat/',
                  help="Name of Directory containing spillover models default= %default")
parser.add_option( "--receiver-models",default='/var/kat/katconfig/user/receiver-models/mkat/',
                  help="Name of Directory containing receiver models default= %default")
parser.add_option( "--nd-models",default='/var/kat/katconfig/user/noise-diode-models/mkat/',
                  help="Name of Dir containing noise diode models models default= %default")

parser.add_option( "--aperture-efficiency",default='/var/kat/katconfig/user/aperture-efficiency/mkat/',
                  help="Name of Directory containing aperture-efficiency models default= %default")

parser.add_option( "--fix-opacity",action="store_true", default=False,
                  help="The opacity is fixed to  0.01078 (Van Zee et al.,1997) or it is calculated according to ITU-R P.676-9.")
parser.add_option("-c", "--channel-mask", default=None,
                  help="Optional pickle file with boolean array specifying channels to mask (default is the "
                  "default mask for the band) if no mask is wanted then specify --channel-mask=''  ")
(opts, args) = parser.parse_args()

if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()

select_freq= np.array(opts.select_freq.split(','),dtype=float)
select_el = np.array(opts.select_el.split(','),dtype=float)
h5 = katdal.open(args[0])
h5.select(scans='track')
nd_models = opts.nd_models
spill_over_models =  opts.spill_over_models
filename = args[0]
channel_bw = opts.freq_bw
freq_bw = opts.freq_bw
#channel_mask = opts.channel_mask #'/var/kat/katsdpscripts/RTS/rfi_mask.pickle'
n_chans = h5.shape[1]

fix_opacity = opts.fix_opacity
freq_chans = opts.freq_chans

for ant in h5.ants:
    #Load the data file
    rec = h5.receivers[ant.name]
    nice_filename =  args[0].split('/')[-1]+ '_' +ant.name+'_tipping_curve'
    pp =PdfPages(nice_filename+'.pdf')
    nice_title = " %s  Ant=%s"%(args[0].split('/')[-1], ant.name)

    # if defined us file specs, otherwise set L-band params
    if ( rec.split('.')[0] != 'undefined' ):
        Band,SN = h5.receivers.get(ant.name,'l.4').split('.') # A safe Default
    else:
        Band = 'L'
        SN = h5.sensor['Antennas/'+ant.name+'/rsc_rxl_serial_number'][0] # Try get the serial no. only used for noise&recever model
        warnings.warn('Warning: Failed to find Receiver model, setting band to L  ')
        print('Warning: Failed to find Receiver model, setting band to L ')

    if opts.channel_mask is None: # use band appopriate masks. #'/var/kat/katsdpscripts/RTS/rfi_mask.pickle'
        if  Band.upper() == 'U' :
            channel_mask =  "/var/kat/katsdpscripts/RTS/rfi_mask_UHF.pickle"
        elif Band.upper() == 'L' :
            channel_mask =  "/var/kat/katsdpscripts/RTS/rfi_mask.pickle"
        else :
            warnings.warn('Warning: Failed to have a Receiver channel_mask for "%s" band setting channel_mask to nothing  '%(Band))
            print('Warning: Failed to have a Receiver channel_mask for "%s" band setting channel_mask to None  '%(Band))
            channel_mask = ''
    else :
        channel_mask =   opts.channel_mask

    print("Channel Mask :%s , Channel Band %s "%(channel_mask,Band))
    receiver_model_H = str("{}/Rx{}_SN{:0>4d}_calculated_noise_H_chan.dat".format(opts.receiver_models,str.upper(Band),int(SN)))
    receiver_model_V = str("{}/Rx{}_SN{:0>4d}_calculated_noise_V_chan.dat".format(opts.receiver_models,str.upper(Band),int(SN)))
    aperture_efficiency_h = "%s/ant_eff_%s_H_AsBuilt.csv"%(opts.aperture_efficiency,str.upper(Band))
    aperture_efficiency_v = "%s/ant_eff_%s_V_AsBuilt.csv"%(opts.aperture_efficiency,str.upper(Band))
    aperture_efficiency = aperture_efficiency_models(filenameH=aperture_efficiency_h,filenameV=aperture_efficiency_v)
    spill_over_model_path = "%s/MK_%s_Tspill_AsBuilt_atm_mask.dat"%(spill_over_models,str.upper(Band))
    SpillOver = Spill_Temp(filename=spill_over_model_path) #/var/kat/katconfig/user/spillover-models/mkat/MK_L_Tspill_AsBuilt_atm_mask.dat

    num_channels = np.int(channel_bw/(h5.channel_width/1e6)) #number of channels per band
    chunks=[h5.channels[x:x+num_channels] for x in xrange(0, len(h5.channels), num_channels)]

    print("Selecting channel data to form %f MHz Channels"%(channel_bw) )
    d = load_cal(filename, "%s" % (ant.name,), nd_models, chunks,channel_mask=channel_mask,n_chan=n_chans,channel_range=freq_chans,band_input=Band.lower())

    tsys = np.zeros((len(d.scans),len(d.freqs[:]),5))#*np.NaN
    tant = np.zeros((len(d.scans),len(d.freqs[:]),5))#*np.NaN

    receiver = Rec_Temp(receiver_model_H, receiver_model_V)
    elevation = np.array([np.average(scan_el) for scan_el in scape.extract_scan_data(d.scans,'el').data])
    ra        = np.array([np.average(scan_ra) for scan_ra in scape.extract_scan_data(d.scans,'ra').data])
    dec       = np.array([np.average(scan_dec) for scan_dec in scape.extract_scan_data(d.scans,'dec').data])
    sort_ind  = elevation.argsort()
    elevation,ra,dec = elevation[sort_ind],ra[sort_ind],dec[sort_ind]
    surface_temperature = np.mean(d.enviro['temperature']['value'])
    air_relative_humidity = h5.humidity.mean()/100. # Fractional
    length = 0
    #freq loop
    for i,freq_val in enumerate(d.freqs):
        if not d is None:
            d.filename = [filename]
            nu = d.freqs  #MHz Centre frequency of observation
            #print("PreLoad T_sysTemp = %.2f Seconds"%(time.time()-time_start))
            T_SysTemp = System_Temp(d,d.freqs[i],freq_index=i,elevation=elevation,ra=ra,dec=dec,surface_temperature = surface_temperature,air_relative_humidity=air_relative_humidity)
            #print("Load T_sysTemp = %.2f Seconds"%(time.time()-time_start))
            units = T_SysTemp.units+''
            fit_H = fit_tipping(T_SysTemp,SpillOver,'HH',d.freqs[i],receiver,fixopacity=fix_opacity)
            #print("Fit tipping H = %.2f Seconds"%(time.time()-time_start))
            fit_V = fit_tipping(T_SysTemp,SpillOver,'VV',d.freqs[i],receiver,fixopacity=fix_opacity)
            #print("Fit tipping V = %.2f Seconds"%(time.time()-time_start))
            #print ('Chi square for HH  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_H['chisq'],))
            #print ('Chi square for VV  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_V['chisq'],))
            length = len(T_SysTemp.elevation)
            Tsky_spec = 2.725 + 1.6*(d.freqs[i]/1e3)**-2.75 # T_SysTemp.Tsys_sky  is Tsys-(Tsky-cmb) . We then add the spec sky aproxx (T_gal+Tcmb)
            tsys[0:length,i,0] = (np.array(T_SysTemp.Tsys_sky['HH'])+Tsky_spec)/aperture_efficiency.eff['HH'](d.freqs[i])
            tsys[0:length,i,1] = (np.array(T_SysTemp.Tsys_sky['VV'])+Tsky_spec)/aperture_efficiency.eff['VV'](d.freqs[i])
            tsys[0:length,i,2] = T_SysTemp.elevation
            tsys[0:length,i,3] = T_SysTemp.sigma_Tsys['HH']/aperture_efficiency.eff['HH'](d.freqs[i])
            tsys[0:length,i,4] = T_SysTemp.sigma_Tsys['VV']/aperture_efficiency.eff['VV'](d.freqs[i])
            tant[0:length,i,0] = np.array(fit_H['fit'])[:,0]
            tant[0:length,i,1] = np.array(fit_V['fit'])[:,0]
            tant[0:length,i,2] = T_SysTemp.elevation
            #print("Debug: T_sys = %f   App_eff = %f  value = %f"%( np.array(fit_H['fit'])[22,0],aperture_efficiency.eff['HH'](d.freqs[i]),np.array(fit_H['fit'])[22,0]/aperture_efficiency.eff['HH'](d.freqs[i])))


    fig = T_SysTemp.sky_fig(freq=freq_val.min())
    fig.savefig(pp,format='pdf')
    plt.close(fig)

    for freq in select_freq :
        title = ""
        if np.abs(d.freqs[:]-freq).min() < freq_bw*1.1 :
            i = (np.abs(d.freqs[:]-freq)).argmin()
            lineval = None
            if str.upper(Band) == 'L':
                if freq > 1420 :
                    lineval = 46
                else:
                    lineval = 42
            fig = plot_data_el(tsys[0:length,i,:],tant[0:length,i,:],title=r"%s $T_{sys}/\eta_{ap}$ and $T_{ant}$ at %.1f MHz"%(nice_title,d.freqs[i]),units=units,line=lineval,aperture_efficiency=aperture_efficiency,frequency=d.freqs[i])
            plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
            fig.savefig(pp,format='pdf')
            plt.close(fig)
    for el in select_el :
        title = ""
        i = (np.abs(tsys[0:length,:,2].max(axis=1)-el)).argmin()
        fig = plot_data_freq(d.freqs[:],tsys[i,:,:],tant[i,:,:],title=r"%s $T_{sys}/\eta_{ap}$ and $T_{ant}$ at %.1f Degrees elevation"%(nice_title,np.abs(tsys[0:length,:,2].max(axis=1))[i]),aperture_efficiency=aperture_efficiency,band=str.upper(Band))
        plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
                #break

    fig = plt.figure(None,figsize = (8,8))
    text =r"""The 'tipping curve' is calculated according to the expression below, with the parameters
of $T_{\mathrm{ant}}$ and $\tau_{0}$, the Antenna temperature and the atmospheric opacity respectively. All the
variables are also functions of frequency.

$T_{\mathrm{sys}}(\mathrm{el}) = T_{\mathrm{cmb}}(\mathrm{ra,dec}) + T_{\mathrm{gal}}(\mathrm{ra,dec}) + T_{\mathrm{atm}}*(1-\exp\left(\frac{-\tau_{0}}{\sin(\mathrm{el})}\right)) + T_{\mathrm{spill}}(\mathrm{el}) + T_{\mathrm{ant}} + T_{\mathrm{rx}}$

$T_{\mathrm{sys}}(\mathrm{el})$ is determined from the noise diode calibration so it is $\frac{T_{\mathrm{sys}}(\mathrm{el})}{\eta_{_{\mathrm{illum}}}}$

We assume the opacity and $T_{\mathrm{ant}}$ is the residual after the tipping curve function is calculated.
$T_{\mathrm{cmb}}$ + $T_{\mathrm{gal}}$ is obtained from the Sky model. """
    if fix_opacity :
        text += r"""$\tau_{0}$, the zenith opacity, is set to 0.01078
(Van Zee et al., 1997). $T_{\mathrm{ant}}$ is the excess temperature since the other components are
known. """
    else:
        text += r"""$\tau_{0}$, the zenith opacity, is the calculated opacity
according to ITU-R P.676-9. $T_{\mathrm{ant}}$ is the excess temperature since the other components are
known. """

    text += r"""The green solid lines in the figures reflect the modelled $T_{\mathrm{sys}}(\mathrm{el})$ or $T_{\mathrm{sys}}(\mathrm{freq})$, with the
broken green lines indicating a $\pm10\%$ margin."""

    params = {'font.size': 10}
    plt.rcParams.update(params)
    ax = fig.add_subplot(111)
    anchored_text = AnchoredText(text, loc=2, frameon=False)
    ax.add_artist(anchored_text)
    ax.set_axis_off()
    plt.subplots_adjust(top=0.99,bottom=0,right=0.975,left=0.01)
    #plt.figtext(0.1,0.1,text,fontsizie=10)
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    fig.savefig(pp,format='pdf')
    pp.close()
    plt.close(fig)
    plt.close('all')




        
