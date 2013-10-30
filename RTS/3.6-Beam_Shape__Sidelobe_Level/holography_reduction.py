import numpy as np
import matplotlib.pyplot as plt
import katfile
from matplotlib.backends.backend_pdf import PdfPages
import optparse
import katholog

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .per10  - 10% percental value in the annulus
          .per25  - 25% percentail value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad)
    radialdata.max = ny.zeros(nrad)
    radialdata.per10 = ny.zeros(nrad)
    radialdata.per25 = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = ny.nan
        radialdata.per10[irad] = ny.nan
        radialdata.per10[irad] = ny.nan
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
        radialdata.per10[irad] = ny.percentile(data[thisindex],10)
        radialdata.per25[irad] = ny.percentile(data[thisindex],25)
    return radialdata


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a tipping curve plot in a pdf file.')
parser.add_option("-f", "--freq-chans", default='200,800',
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default %default)")
parser.add_option("--in",
                  help="Name of directory containing spillover models")
parser.add_option( "--sky-map", default='~/comm/catalogues/TBGAL_CONVL.FITS',
                  help="Name of map of sky tempreture in fits format', default = '%default'")

(opts, args) = parser.parse_args()



if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')

nice_filename =  args[0]+ '_holography'
pp = PdfPages(nice_filename+'.pdf')

interactive = True
dataset=katholog.Dataset("1379565877.h5",'kat7')
freq = [1650]
#TODO add automated rfi flagging 
if interactive: dataset.flagplot()
a = raw_input('Please press a key after completing the Flagging step')


fullbeam=katholog.BeamCube(dataset,freqMHz=freq)
extents=[fullbeam.margin[0],fullbeam.margin[-1],fullbeam.margin[0],fullbeam.margin[-1]]

emssdataset=katholog.Dataset('Dish_1600.pat','kat7emss',freq_MHz=freq,clipextent=40)
emssbeam=katholog.BeamCube(emssdataset,extent=40)

power=np.abs(fullbeam.Gx)**2+np.abs(fullbeam.Gy)**2+np.abs(fullbeam.Dx)**2+np.abs(fullbeam.Dy)**2
power/=np.max(power)

emsspower=np.abs(emssbeam.Gx)**2+np.abs(emssbeam.Gy)**2+np.abs(emssbeam.Dx)**2+np.abs(emssbeam.Dy)**2
emsspower/=np.max(emsspower)

aperturemap=katholog.ApertureMap(dataset)

text = []

if True : # R.RS.P.40 Req
    #radial = radial_data(power[0,:,:])
    radial = radial_data(power[0,:,:],annulus_width=5)
    #TODO  Fix the line below 
    ind_2nd_sidelobe = (np.diff(radial.mean)>0).nonzero()[0][0:-1][np.diff((np.diff(radial.mean)>0).nonzero()[0])>1][1] + 1
    text.append("R.RS.P.40 Max 2nd Sidelevel %f dB at %f Degrees"%(10*np.log10(radial.max[ind_2nd_sidelobe]),np.abs(fullbeam.margin[256+ind_2nd_sidelobe*5])))


    pixel_per_deg = 512./(fullbeam.margin[-1]-fullbeam.margin[0])
    y,x = np.ogrid[-256:256, -256:256]
    mask = x*x + y*y > (10.*pixel_per_deg)**2
    pix_sr = ( (fullbeam.margin[-1]-fullbeam.margin[0])**2/(512.0**2) )/(180/np.pi)**2 #pixel to sr
    text.append("R.RS.P.40 Total amount of sky > 10 degrees for boresight with sidelobes > -40 dB is %f sr."%((10*np.log10(power[0,...]*mask)>-40).sum()*pix_sr))


if True : # R.RC.P.4
    #innerbeam=katholog.BeamCube(dataset,freqMHz=1800,extent=4) #scanantennaname='ant5'
    mask_3dB=10.0*log10(power)>-3
    text.append("R.RC.P.4 Mean varation between model and data within 3 dB area is %f percent"%(((emsspower[mask_3dB]-power[mask_3dB])/emsspower[mask_3dB]).mean()*100))


fig = plt.figure()
jones = fullbeam.plot('Gx','amp')
fig.savefig(pp,format='pdf') 
plt.close(fig)

fig = plt.figure()
jones = fullbeam.plot('Gy','amp')
fig.savefig(pp,format='pdf') 
plt.close(fig)
   
fig = plt.figure()
jones = fullbeam.plot('Gx','phase')
fig.savefig(pp,format='pdf') 
plt.close(fig)

fig = plt.figure()
jones = fullbeam.plot('Gy','phase')
fig.savefig(pp,format='pdf') 
plt.close(fig)



innerbeam=katholog.BeamCube(dataset,freqMHz=freq,extent=4) #scanantennaname='ant5'
centralbeam = innerbeam
mainlobe=np.abs(centralbeam.Gx)**2+np.abs(centralbeam.Gy)**2+np.abs(centralbeam.Dx)**2+np.abs(centralbeam.Dy)**2
mainlobe/=np.max(mainlobe)

if True : # R.RC.P.2 & R.RC.P.3
    innerextents=[innerbeam.margin[0],innerbeam.margin[-1],innerbeam.margin[0],innerbeam.margin[-1]]
    mask_3dB=10.0*log10(mainlobe)>-3  ## 3dB Mask  for   R.RC.P.3
    mask_1dB=10.0*log10(mainlobe)>-1  ## 1dB Mask  for   R.RC.P.2
    dx = (10*np.log10(np.abs(innerbeam.Dx[mask_3dB])**2)>-20).sum()
    dy = (10*np.log10(np.abs(innerbeam.Dy[mask_3dB])**2)>-20).sum()
    text.append("R.RC.P.3 Number of points in Jones Dx^2 matrix in 3dB contour > -20 dB is %i"%(dx))
    text.append("R.RC.P.3 Number of points in Jones Dy^2 matrix in 3dB contour > -20 dB is %i"%(dy))
    dx = (10*np.log10(np.abs(innerbeam.Dx[mask_1dB])**2)>-26).sum()
    dy = (10*np.log10(np.abs(innerbeam.Dy[mask_1dB])**2)>-26).sum()
    text.append("R.RC.P.2 Number of points in Jones Dx^2 matrix in 1dB contour > -26 dB is %i"%(dx))
    text.append("R.RC.P.2 Number of points in Jones Dy^2 matrix in 1dB contour > -26 dB is %i"%(dy))
    fig = plt.figure()
    plt.imshow(np.abs(10*np.log10(innerbeam.Dx[0,...])*mask_3dB[0,...]),extent=innerextents)
    plt.colorbar()
    plt.title('Jones Dx Matrix (amplitude)')
    plt.xlabel('Degrees')
    plt.ylabel('Degrees')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)
    fig = plt.figure()
    plt.imshow(np.abs(10*np.log10(innerbeam.Dy[0,...])*mask_3dB[0,...]),extent=innerextents)
    plt.colorbar()
    plt.xlabel('Degrees')
    plt.ylabel('Degrees')
    plt.title('Jones Dy Matrix (amplitude)')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)
   



#crosscoupling_x_feed_in_mainlobe=np.max(10*np.log10(np.abs(centralbeam.Dx.reshape(-1)[mainlobeind])**2))
##crosscoupling_x_feed_in_mainlobe=np.max(10*np.log10(np.abs(centralbeam.Dx.reshape(-1)[mainlobeind])**2/np.abs(centralbeam.Gx.reshape(-1)[mainlobeind])))



#plt.figure()
#fullbeam.plot('jones')
if True : #R.A.P.3 & R.T.P.98
    plt.figure()
    aperturemap.plot('amp')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    plt.figure()
    aperturemap.plot('phase')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    plt.figure()
    aperturemap.plot('unwrap')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    plt.figure()
    aperturemap.plot('model')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    plt.figure()
    aperturemap.plot('dev')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    plt.figure()
    aperturemap.plot('flat')
    fig.savefig(pp,format='pdf') 
    plt.close(fig)

    text.append("R.T.P.98  Surface roughness is %f mm RMS unwraped or  %f mm RMS flaterned  "%(aperturemap.rms0_mm,aperturemap.rms_mm))
    text.append('Measured gain with observed illumination: %.2f dB'%(aperturemap.gainmeasured_dB))
    text.append('Theorectical gain with uniform illumination: %.2f dB'%(aperturemap.gainuniform_dB))
    text.append('Gain with no panel errors: %.2f dB'%(aperturemap.gainnopanelerr_dB))
    text.append('Gain with only feed offset: %.2f dB'%(aperturemap.gainmodel_dB))
    text.append('Aperture efficiency: %f'%(aperturemap.eff_aperture))
    text.append('Illumination efficiency: %f'%(aperturemap.eff_illumination))
    text.append('Taper efficiency: %.3f'%(aperturemap.eff_taper))
    text.append('Phase efficiency: %.3f'%(aperturemap.eff_phase))
    text.append('Spillover efficiency: %.3f'%(aperturemap.eff_spillover))
    text.append('Surface-error efficiency: %.3f'%(aperturemap.eff_surface))

fig = plt.figure(None,figsize = (10,16))
plt.figtext(0.1,0.1,'\n'.join(text),fontsize=10)
fig.savefig(pp,format='pdf')
pp.close()
plt.close(fig)

