"""
    
"""

import numpy as np
import katdal
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_pdf import PdfPages
import optparse

def Ang_Separation(pos1,pos2):
    Ra1 = pos1[0]
    Dec1 = pos1[1]
    Ra2 = np.array(pos2[0,:])
    Dec2 = np.array(pos2[1,:])
    top = np.cos(Dec2)**2*np.sin(Ra2-Ra1)**2+(np.cos(Dec1)*np.sin(Dec2)-np.sin(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1))**2
    bottom = np.sin(Dec1)*np.sin(Dec2)+np.cos(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1)
    return np.arctan2(np.sqrt(top),(bottom))

def gauss2d(x,y,x0,y0,xfwhm,yfwhm,height):
    """ Calculate a 2d gaussian."""
    const2 =  2.*np.sqrt(2.*np.log(2.))
    xsig = xfwhm / const2
    ysig = yfwhm / const2
    term1 = (x-x0)**2 / (2*xsig*xsig)
    term2 = (y-y0)**2 / (2*ysig*ysig)
    return height * np.exp(-(term1 + term2))

def gauss1d(x,xfwhm):
    """ Calculate a 2d gaussian."""
    xsig = xfwhm / (2.*np.sqrt(2.*np.log(2.)))
    term1 = (x)**2 / (2*xsig*xsig)
    return (1./(xsig*np.sqrt(2.*np.pi)) )*np.exp(-term1)

def beamwidth(fwhm): return fwhm / (2.*np.sqrt(2.*np.log(2.)))


c = 299792458 # 3e8 # m/s, speed of light in vacuum

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period


def dif(func,x,xfwhm):
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.mean(np.abs(x)))
    return (func(x + eps,xfwhm) - func(x - eps,xfwhm)) / (2.0 * eps )

def getper(x,c=50):
    a = np.percentile(np.real(x),c)
    sign = np.sign(a)
    return sign*np.sqrt(np.abs(a))*3600



# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a Jitter plot in a pdf file.')
parser.add_option( "--bins", default=40,
                  help="The nuber of bins to use when evaluation the different seperations', default = '%default'")
parser.add_option( "--ant", default='ant4',
                  help="The antenna to do the reduction for', default = '%default'")

(opts, args) = parser.parse_args()



if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')
height = 1.0
bins = opts.bins


h5 = katdal.open(args[0])
#1392246099.h5

nice_filename =  args[0].split('/')[-1]+ '_' +opts.ant+'_jitter_test'
pp = PdfPages(nice_filename+'.pdf')

h5.select(scans='track',ants=opts.ant,channels=slice((h5.channels.shape[0])//4,(h5.channels.shape[0]*3)//4))
pos1,pos2 = np.radians((h5.az[:,0],h5.el[:,0])),np.array(h5.catalogue.targets[1].azel(h5.timestamps[:]))


hpbw = fwhm = np.degrees(1.02*(c/h5.channel_freqs)/h5.ants[0].diameter)
pos1,pos2 = np.radians((h5.az[:,0],h5.el[:,0])),np.array(h5.catalogue.targets[1].azel(h5.timestamps[:]))
sep = np.degrees(Ang_Separation(pos1,pos2))
hist,binvals = np.histogram(sep,bins=bins)
binvals[-1] = binvals[-1] + 0.1
digibins = np.digitize(sep,bins=binvals)


var = np.zeros((np.array(hist.nonzero()[0]).shape[0],h5.channels.shape[0],h5.shape[-1]))
mean  = np.zeros((np.array(hist.nonzero()[0]).shape[0],h5.channels.shape[0],h5.shape[-1]))
baseline_mean = (np.mean(h5.vis[digibins == digibins.max(),:,:],axis=0))# Off source Mean
peak_mean = (np.mean(h5.vis[digibins == digibins.min(),:,:],axis=0))# On source Mean
norm = np.abs(1./(peak_mean-baseline_mean) *np.sqrt(2.*np.pi)*beamwidth(fwhm)[np.newaxis,:,np.newaxis])
var_inf = (np.std((h5.vis[digibins == digibins.max(),:,:]-baseline_mean)*norm,axis=0))**2  # Off source variance

returntext = []
for blcount,blvalue in enumerate(h5.corr_products[:]) :
    if  h5.corr_products[blcount,0] == h5.corr_products[blcount,1]:
        upper = []
        lower = []
        mean =  []
        thetav = []
        returntext.append('Calculated Antenna short timescale jitter for %s'%(blvalue[0]))
        returntext.append('Antenna, Angle ,  mean ,  lower ,upper errors')
        for n,i in enumerate(hist.nonzero()[0]) :
            data =  np.abs((h5.vis[digibins == i+1,:,:]-baseline_mean)*norm)
            var[n,:,:] = (np.std(data,axis=0))**2 - var_inf #  digitize has a [1..bins] index
            var_amp = var[n,:,blcount]
            theta = sep[digibins == i+1].mean()
            if theta > 0.01 and theta < fwhm.max() :
                #var_0 = 1./(2.*np.pi*beamwidth(fwhm)**2*(-np.log(var_amp*beamwidth(fwhm)**6)))
                var_theta = var_amp*2.*np.pi*beamwidth(fwhm)**6*(1./theta**2)*np.exp(theta**2/beamwidth(fwhm)**2)   
                thetav.append(theta)
                lower.append(getper(var_theta,c=50.-34.13))
                mean.append(getper(var_theta,c=50))
                upper.append(getper(var_theta,c=50.+34.13))
                returntext.append('%s, %.4f ,  %.2f ,  %.2f ,  %.2f'%(blvalue[0],theta,getper(var_theta,c=50),getper(var_theta,c=50.-34.13),getper(var_theta,c=50.+34.13)))
        mean = np.array(mean)
        lower = np.array(lower)
        upper = np.array(upper)
        fig = plt.figure(None)
        plt.title('Calculated Antenna short timescale jitter for %s'%(blvalue[0]))
        plt.xlabel("Angle offset from Boresight (degrees)")
        plt.ylabel("Standard Devation of Telescope pointing (arcseconds)")
        #plt.plot(thetav,mean,'go')
        #plt.plot(thetav,lower,'ro')
        #plt.plot(thetav,upper,'bo')
        plt.errorbar(thetav,mean, yerr=(mean-lower,upper-mean) )
        # the formulate is valid in these ranges 0.25 -> 0.55  
        fig.savefig(pp,format='pdf') 
        plt.close(fig)

        
        
fig = plt.figure(None,figsize = (10,16))
plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
fig.savefig(pp,format='pdf')
pp.close()
plt.close(fig)


