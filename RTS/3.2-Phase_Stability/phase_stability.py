#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration
 
import numpy as np
import matplotlib.pyplot as plt   
import optparse
import katfile
from matplotlib.backends.backend_pdf import PdfPages
import stefcal


def cplot(data,*args,**kwargs):
    if data.dtype.kind == 'c': plot(np.real(data),np.imag(data),*args,**kwargs)
    else : plot(data,*args,**kwargs)

def mean(a,axis=None):
    """This function calclates the mean along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the mean of the argument & the angle (exp(1j*theta) )
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    >>>>mean(np.array((1+1j , 1-1j)))
    (1.4142135623730951+0j)
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
        th = np.arctan2(a.imag,a.real)
        sa = (np.sin(th)).sum(axis=axis)
        ca = (np.cos(th)).sum(axis=axis)
        thme = np.arctan2(sa,ca)
        return r*np.exp(1j*thme)
    else:
        return np.mean(a,axis=axis)


def yamartino_method(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation of the argument & the angle (exp(1j*theta) )
    This uses the Yamartino method which is a one pass method of estimating the standard devation 
    of an angle
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#mean
        th = np.arctan2(a.imag,a.real)
        if axis is None :
            sa = (np.sin(th)/len(th)).sum()
            ca = (np.cos(th)/len(th)).sum()
        else:
            sa = (np.sin(th)/len(th)).sum(axis=axis)
            ca = (np.cos(th)/len(th)).sum(axis=axis)
        e = np.sqrt(1.-(sa**2+ca**2))
        thsd = np.arcsin(e)*(1.+(2./np.sqrt(3)-1.)*e**3)
        return r*np.exp(1j*thsd)
    else:
        return np.std(a,axis=axis)

def std(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation of the argument & the angle (exp(1j*theta) )
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
    The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#mean
        th = np.arctan2(a.imag,a.real)
        sa = np.sin(th).sum(axis=axis)
        ca = np.cos(th).sum(axis=axis)
        thme = np.arctan2(sa,ca)        
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1-np.cos(th-np.reshape(thme,nshape)).mean(axis=axis)
        return r*np.exp(1j*np.sqrt(-2.*np.log(1.-S0)))
    else:
        return np.std(a,axis=axis)


def absstd(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation  of the angle (exp(1j*theta) )
    and standard devation over the mean of the argument
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        rstd = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#std
        rmean = np.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
        th = np.arctan2(a.imag,a.real)
        sa = np.sin(th).sum(axis=axis)
        ca = np.cos(th).sum(axis=axis)
        thme = np.arctan2(sa,ca)        
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1-np.cos(th-np.reshape(thme,nshape)).mean(axis=axis)
        return (rstd/rmean)*np.exp(1j*np.sqrt(-2.*np.log(1.-S0)))
    else:
        return np.std(a,axis=axis)/np.mean(a,axis=axis)

def plot_AntennaGain(gains,freq,inputs):
    """ This plots the Amplitude and Phase across the bandpass
    gains   : complex array shape (time,frequency,antenna) or (frequency,antenna)  
    freq    : array shape (frequency) in Herts
    inputs  : list of string names of the antennas
    returns : a list of figure objects
    """
    returned_plots = []
    if len(gains.shape) == 2 : gains =gains[np.newaxis,:,:]
    for i in xrange(gains.shape[-1]):
        fig, (ax) = plt.subplots(nrows=2, sharex=True)
        ax[0].set_title('Phase %s'%(inputs[i]))
        ax[1].set_title('Amplitude %s'%(inputs[i]))
        ax[0].set_ylabel('Degrees')
        ax[1].set_xlabel("Frequency (MHz)")
        ax[0].plot(freq/1e6,np.degrees(np.angle(gains[:,:,i].T)))
        ax[1].plot(freq/1e6,np.abs(gains[:,:,i].T))
        returned_plots.append(fig)
    return returned_plots

def plot_DataStd(data,freq,corr_products,log_amplitude=True):
    """ This plots the Amplitude and Phase across the bandpass
    gains         : complex array shape (time,frequency,corrprod) or (time,frequency,corrprod+corrprod.conj())
    freq          : array shape (frequency) in Herts
    corr_products : list of string names of the correlation pair or antenna
    returns       : a list of figure objects
    """
    returned_plots = []
    corr_products = np.array(corr_products)
    if len(np.shape(corr_products)) == 2 : corr_products = np.array(["%s x %s" %(c1,c2) for c1,c2 in corr_products])
    for i in xrange(corr_products.shape[0]):
        tmpdata = std(data[:,:,i],axis=0) # ease up the memory usage
        fig, (ax) = plt.subplots(nrows=2, sharex=True)
        ax[0].set_title('Circular Standard Devation of the Phase -: %s'%(corr_products[i]))
        ax[1].set_title('Standard Devation of the Amplitude -: %s'%(corr_products[i]))
        ax[0].set_ylabel('Degrees')
        ax[1].set_xlabel("Frequency (MHz)")
        if log_amplitude : ax[1].set_yscale('log')
        ax[0].plot(freq/1e6,np.degrees(np.angle(tmpdata)))
        ax[1].plot(freq/1e6,np.abs(tmpdata))
        returned_plots.append(fig)
    return returned_plots



def  fringe_stopping(data):
    new_ants = {
    'ant1' : ('25.0950 -9.0950 0.0450', 23220.506e-9, 23228.551e-9),
    'ant2' : ('90.2844 26.3804 -0.22636', 23283.799e-9, 23286.823e-9),
    'ant3' : ('3.98474 26.8929 0.0004046', 23407.970e-9, 23400.221e-9),
    'ant4' : ('-21.6053 25.4936 0.018615', 23514.801e-9, 23514.801e-9),
    'ant5' : ('-38.2720 -2.5917 0.391362', 23676.033e-9, 23668.223e-9),
    'ant6' : ('-61.5945 -79.6989 0.701598', 23782.854e-9, 23782.150e-9),
    'ant7' : ('-87.9881 75.7543 0.138305', 24047.672e-9, 24039.237e-9),}
    delays = {}
    for inp in data.inputs:
        ant, pol = inp[:-1], inp[-1]
        delays[inp] = new_ants[ant][1 if pol == 'h' else 2]
    center_freqs = data.channel_freqs
    wavelengths = 3.0e8 / center_freqs
    # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
    cable_delay_turns = np.array([(delays[inpB] - delays[inpA]) * center_freqs for inpA, inpB in data.corr_products]).T
    crosscorr = [(data.inputs.index(inpA), data.inputs.index(inpB)) for inpA, inpB in data.corr_products]
    # Assemble fringe-stopped visibility data for main (bandpass) calibrator
    vis_set = None
    for compscan_no,compscan_label,target in data.compscans():
        print "loop",compscan_no,compscan_label,target
        vis = data.vis[:,:,:]
        # Number of turns of phase that signal B is behind signal A due to geometric delay
        geom_delay_turns = - data.w[:, np.newaxis, :] / wavelengths[:, np.newaxis]
        # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
        vis *= np.exp(2j * np.pi * (geom_delay_turns + cable_delay_turns))
        if vis_set is None:
            vis_set = vis.copy()
        else:
            vis_set = np.append(vis_set,vis,axis = 0)
    return vis_set




def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kargs):
    """
     This function produces a rolling window shaped data with the rolled data in the last col
        a      :  n-D array of data  
        window : integer is the window size
        axis   : integer, axis to move the window over
                 default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
                        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
                        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
                      con  vector along each axis.
        'median'        Pads with the median value of all or part of the
                        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
                        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
                        the first and last values of the vector along each
                        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
                        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
                        The first values are used to pad the end and the
                        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
                        see numpy.pad notes
        **kargs are passed to the function numpy.pad
        
    Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis
        
    Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
    """
    if axis == -1 : axis = len(a.shape)-1 
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis: 
                pad_width += [(window//2,window//2 -1 +mod(window,2))]
            else :  
                pad_width += [(0,0)] 
        a = np.pad(a,pad_width=pad_width,mode=mode,**kargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to 

def c_str(a,figures=4):
    if np.iscomplex(a) : return "($Arg = %s  $,$ \phi = %s\degree$)"%(np.round(np.abs(a),figures),np.round(np.degrees(np.angle(a)),figures) )
    return str(a)

def calc_stats(timestamps,data,freq,pol='no polarizarion',windowtime=1200):
    """ calculate the Stats needed to evaluate the obsevation"""
    returntext = []
   
    time = timestamps.max() - timestamps.min()
    time_step = time/timestamps.shape[0]
    window = int(round(windowtime/time_step) )
    for i in xrange(data.shape[-1]):
        windowgainchange = absstd(rolling_window(data, window,axis=-2),axis=-1)
        returntext.append("Total time of obsevation : %f (seconds) with %i accumulations."%(time,timestamps.shape[0]))
        returntext.append("The mean gain of %s is: %.5f"%(pol,mean(data)))
        returntext.append("The Std. dev of the gain of %s is: %.5f"%(pol,gain.std()))
        returntext.append("The RMS of the gain of %s is : %.5f"%(pol,rms))
        returntext.append("The variation of %s is: %.5f"%(pol, c_str(absstd(data))))
        returntext.append("The mean variation over %i samples of %s is: %.5f"%(window,pol,mean(windowgainchange)))
        returntext.append("The max variation over %i samples of %s is: %.5f"%(window,pol,np.max(windowgainchange))) 
        returntext.append("The Peak to Peak  average over %i samples of %s is: %.5f"%(window,pol,np.mean(np.ptp(windowgainchange))))
        returntext.append("The Max Peak to Peak over %i sample set for the obsevation of %s is: %.5f"%(window,pol,np.max(np.ptp(windowgainchange))))
    return returntext  # a plot would be cool


# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs decribing the gain sability for each antenna in the file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='200,800',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')
    


# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])



#h5 = katfile.open(args[0])
h5 = katfile.open('1363727516.h5')
h5.select(corrprods='cross')
# loop over both polarisations
h5.select(ants='ant1,ant2,ant3,ant4,ant5',pol='h',corrprods='cross',scans='track',dumps=slice(1,500))
if np.all(h5.sensor['DBE/auto-delay'] == '0') :
    vis = fringe_stopping(h5)
else:
    vis = h5.vis

flaglist = ~h5.flags()[:,:,:].any(axis=0).any(axis=-1)
flaglist[0:200] = False
flaglist[800:1024] = False
antA = [h5.inputs.index(inpA) for inpA, inpB in h5.corr_products]
antB = [h5.inputs.index(inpB) for inpA, inpB in h5.corr_products]

N_ants = len(h5.ants)
#full_vis = np.concatenate((vis, vis.conj()), axis=-1)
full_antA = np.r_[antA, antB]
full_antB = np.r_[antB, antA]

weights= np.abs(1./np.angle(absstd(vis[:,:,:],axis=0)))
weights= np.concatenate((weights, weights), axis=-1)

# use vector mean == np.mean  on visabilitys
# but use angle mean on solutions/phase change.
gains = stefcal.stefcal( np.concatenate( (np.mean(vis,axis=0), np.mean(vis.conj(),axis=0)), axis=-1) , N_ants, full_antA, full_antB, num_iters=50,weights=weights)
calfac = 1./(gains[np.newaxis][:,:,full_antA]*gains[np.newaxis][:,:,full_antB].conj())

h5.select(ants='ant1,ant2,ant3,ant4,ant5',pol='h',corrprods='cross',scans='track')
data = np.zeros((h5.shape[0:3:2]),dtype=np.complex)
i = 0
for scan in h5.scans():
    print scan
    if np.all(h5.sensor['DBE/auto-delay'] == '0') :
        print "stopping fringes for size ",h5.shape
        vis = fringe_stopping(h5)
    else:
        vis = h5.vis
    data[i:i+h5.shape[0]] = mean((vis*calfac[:,:,:h5.shape[-1]])[:,flaglist,:],axis=1)
    i += h5.shape[0]
    break



#figlist = plot_AntennaGain(gains,h5.channel_freqs,h5.inputs)
#figlist += plot_DataStd(gains,freq,h5.inputs)
#figlist += plot_DataStd(vis,freq,h5.corr_products)
figlist = []
nice_filename =  args[0]+ '_phase_stability'
pp = PdfPages(nice_filename+'.pdf')
for fig in figlist:
    fig = plot_figures(d_uncal, d_cal, time, gain_hh, 'HH')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

returntext = calc_stats(d.timestamps,g_hh,d.freqs,'HH',1200)+calc_stats(d.timestamps,g_vv,d.freqs,'VV',1200)
fig = plt.figure(None,figsize = (10,16))
plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
fig.savefig(pp,format='pdf')
pp.close()
plt.close(fig)


#data = mean(rolling_window(mean(vis[:,flaglist,:]*calfac[:,flaglist,:calfac.shape[-1]//2],axis=1),50,axis=0),axis=-1)



for i in xrange(h5.shape[0]) :plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[i,200:800,0][0,:,0]),'b.',alpha=0.1)
plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0)),'g',)
plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0))+3*np.abs(h5.vis[:,200:800,0].std(axis=0)),'r')
plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0))-3*np.abs(h5.vis[:,200:800,0].std(axis=0)),'r')
