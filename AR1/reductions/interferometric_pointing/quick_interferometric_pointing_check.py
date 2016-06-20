#!/usr/bin/python
# Script that uses katsdpcal's calprocs to reduce data consisting of offset tracks on multiple point sources.
#

from katsdpcal import calprocs
import pickle
import katdal
import numpy as np
import scikits.fitting as fit
import katpoint
import optparse


def plot_data(x,y,z,gsize=100,scatter=False,title=''):
    """Plotting function
    This plots a rasterscan as an intensity image
    the scatter parameter adds markers to indecate the data points
    x,y,z must be the same length and z is the amplitude"""
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    # define grid.
    npts = z.shape[0]
    xi = np.linspace(x.min(),x.max(),gsize)
    yi = np.linspace(y.min(),y.max(),gsize)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]))
    # contour the gridded data, plotting dots at the randomly spaced data points.
    plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    plt.contourf(xi,yi,zi,100,cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    # plot data points.
    if scatter :
        plt.scatter(x,y,marker='o',c='b',s=5)
    plt.xlim(x.min(),x.max())
    plt.ylim(y.min(),y.max())
    plt.title('%s (%d points)' %(title,npts))

def activity(h5,state = 'track'):
    """Activity Sensor because some of antennas have a mind of their own, 
    others appear to have lost theirs entirely """
    antlist = [a.name for a in h5.ants]
    activityV = np.zeros((len(antlist),h5.shape[0]) ,dtype=np.bool)
    for i,ant in enumerate(antlist) :
        sensor = h5.sensor['Antennas/%s/activity'%(ant)]
        activityV[i,:] +=   (sensor==state)
    return np.all(activityV,axis=0)

def interp_sensor(h5, quantity, default,degree=0):
    """Interpolate environmental sensor data."""
    try:
        sensor = h5.sensor[quantity] # Hack for incomplete katdalV3
    except KeyError:
        return (lambda times: default)
    else:
        interp = fit.PiecewisePolynomial1DFit(max_degree=degree)
        interp.fit(sensor['timestamp'],np.array(sensor['value']).astype(np.float))
        return interp

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

def Ang_Separation(pos1,pos2):
    """Calculate the greatest circle distance between po1 and pos2[.....] in radians  """
    Ra1 = pos1[0]
    Dec1 = pos1[1]
    Ra2 = np.array(pos2[0,:])
    Dec2 = np.array(pos2[1,:])
    top = np.cos(Dec2)**2*np.sin(Ra2-Ra1)**2+(np.cos(Dec1)*np.sin(Dec2)-np.sin(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1))**2
    bottom = np.sin(Dec1)*np.sin(Dec2)+np.cos(Dec1)*np.cos(Dec2)*np.cos(Ra2-Ra1)
    return np.arctan2(np.sqrt(top),(bottom))


def compscan_radec_offset(h5 ,channel_mask = '/var/kat/katsdpscripts/RTS/rfi_mask.pickle',chunks=16):
    """Break the band up into chunks"""
    import pickle
    chunk_size = chunks
    rfi_static_flags = np.tile(False, h5.shape[0])
    target = h5.catalogue.targets[h5.target_indices[0]]
    if len(channel_mask)>0:
        pickle_file = open(channel_mask)
        rfi_static_flags = pickle.load(pickle_file)
        pickle_file.close()
    gains_p = {}
    stdv = {}
    avg= np.zeros((chunk_size* 2,10,len(h5.ants)) ) # freq Chunks * pol , pos *8 , Ant     
    h5.antlist = [a.name for a in h5.ants]
    h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
    pols = ["H","V"] # Put in logic for Intensity
    for i,pol in enumerate(pols) :
        gains_p[pol] = []
        pos = []
        stdv[pol] = []
        h5.select(pol=pol)
        h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
        for scan in h5.scans() : 
            track_ind = activity(h5,state = 'track')
            data = h5.vis[track_ind]
            if data.shape[0] > 1 :
                gains_p[pol].append(calprocs.g_fit(data[:,:,:].mean(axis=0),h5.bls_lookup,refant=0) )
                stdv[pol].append(np.ones((data.shape[0],data.shape[1],len(h5.ants))).sum(axis=0))#number of data points
                pos.append( [h5.ra[track_ind,:].mean(axis=0), h5.dec[track_ind,:].mean(axis=0)] ) # time,ant
        for ant in xrange(len(h5.ants)):
            for chunk in xrange(chunks):
                if np.array(pos).shape[0] > 1 : # a good proxy for data 
                    freq = slice(chunk*256,(chunk+1)*256)
                    rfi = ~rfi_static_flags[freq]   
                    fitobj  = fit.GaussianFit(np.array(pos)[:,:,ant].mean(axis=0),[1,1],1) 
                    x = np.column_stack((np.array(pos)[:,0,ant],np.array(pos)[:,1,ant]))
                    y = np.abs(np.array(gains_p[pol])[:,freq,:][:,rfi,ant]).mean(axis=1)
                    y_err = 1./np.sqrt(np.array(stdv[pol])[:,freq,:][:,rfi,ant].sum(axis=1))
                    gaussian = fitobj.fit(x.T,y,y_err )
                    
                    avg[chunk+i*chunk_size,0:2,ant] = gaussian.mean
                    avg[chunk+i*chunk_size,2:4,ant] = gaussian.std_mean
                    avg[chunk+i*chunk_size,4:6,ant] = gaussian.std
                    avg[chunk+i*chunk_size,6:8,ant] = gaussian.std_std
                    avg[chunk+i*chunk_size,8,ant] = gaussian.height
                    avg[chunk+i*chunk_size,9,ant] = gaussian.std_height
    return avg



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it and calculates the offset from the target"
                                           "postion.")

parser.add_option("-c", "--channel-mask", default="/var/kat/katsdpscripts/RTS/rfi_mask.pickle", help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
#parser.add_option("-o", "--output", dest="outfilebase",default=None,
#                  help="Base name of output files (*.csv for output data and *.log for messages, "
#                       "default is '<dataset_name>_interferometric_pointing')")

(opts, args) = parser.parse_args()

if len(args) != 1 or not args[0].endswith('.h5'):
    raise RuntimeError('Please specify a single HDF5 file as argument to the script')

channel_mask = opts.channel_mask #

h5 = katdal.open(args)  # THis is an old KAT-7 file with no fringestopping
h5.select(compscans='interferometric_pointing')
h5.antlist = [a.name for a in h5.ants]
h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
#if opts.outfilebase is None :
#    outfilebase =  "%s_%s"%(h5.name.split('/')[-1].split('.')[0], "interferometric_pointing")
#else:
#    outfilebase = opts.outfilebase
for cscan in h5.compscans() :
    avg = compscan_radec_offset(h5,channel_mask)
    print "Compound scan %i of field %s "%(cscan[0],cscan[2].name)
    if len(avg) > 0 : # if not an empty set
        pos1 = np.average(avg[:,0:2,:],axis=0,weights=1./avg[:,2:4,:]**2)
        for ant in xrange(len(h5.ants)):
            sep = np.degrees(Ang_Separation(cscan[2].radec(),np.radians(pos1)) )
            print "%s  : %s  Separation=%2.2f' "%(h5.ants[ant].name,cscan[2].name,sep[ant]*60)
