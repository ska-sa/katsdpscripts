from optparse import OptionParser

import os
import random

import numpy as np
import matplotlib.pyplot as plt

def sign(number):
    if str(number)[0]=='-': return -1
    else: return 1
def DDMMSS2DEG(coord_vec):
    if len(coord_vec) ==1 : return float(coord_vec)
    return sign(float(coord_vec[0]))*(np.abs(float(coord_vec[0])) + float(coord_vec[1])/60.0 + float(coord_vec[2])/3600.0)

def readfile(filename):
    try:
        fin = open(filename)
    except IOError: raise
    data = fin.readline()
    fin.close()

    return (data, [DDMMSS2DEG(np.array(coeff.split(':'),dtype=float)) for coeff in data.strip().split(',')])

def selectedcoverage():
    skypoints = []
    for elevation in [5,45,85]:
        for azimuth in [0, 90, 180, 270]:
            skypoints.append([float(azimuth), float(elevation)])
    return skypoints

def fullskycoverage():
    skypoints = []
    for elevation in range(15,88):
        for azimuth in range(0,359):
            skypoints.append([float(azimuth), float(elevation)])
    return skypoints

def fakeskycoverage(nrpoints):
    skypoints = []
    for point in range(nrpoints):
        elevation=random.randrange(15,89)
        azimuth=random.randrange(0,360)
        skypoints.append([float(azimuth), float(elevation)])
    return skypoints

def getbeamrotation(Az, El, p3, p4, p5, p6):
    return np.array((p4*np.sin(np.deg2rad(El))+p3+(p5*np.sin(np.deg2rad(Az))-p6*np.cos(np.deg2rad(Az))))*np.tan(np.deg2rad(El)))

if __name__ == '__main__':

    # python beamrotation.py
    # --ants='m001,m003,m006,m007,m008,m010,m014,m015,m021,m022,m024,m025,m034,m036,m062,m063'
    # katconfig/mkat/*.l.pm.csv --full

    # python beamrotation.py katconfig/user/pointing-models/mkat/*.l.pm.csv --debug

    usage = " \
              \npython %prog [options] <antenna_pointing_model_file(s)> \
              \nExamples: \
              \npython %prog  \
              \n--ants='m001,m003,m006,m007,m008,m010,m014,m015,m021,m022,m024,m025,m034,m036,m062,m063' \
              \nkatconfig/mkat/*.l.pm.csv --full \
            "
    parser = OptionParser(usage=usage, version="%prog 1.0")
    parser.add_option('-n', '--npoints', dest='npoints', type=int, default=None,
                      help='Sky coverage simulated using npoint random samples of (Az,El) points')
    parser.add_option('--ants', dest='ants', type=str, default=None,
                      help='List of antennas to evaluate')

    parser.add_option('--full', dest='full', action='store_true', default=False,
                      help='Get points over full sky coverage')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False,
                      help='Show various additional graphs and output results')
    parser.add_option('--debug',         dest='debug', action='store_true', default=False,
                      help='Show example output for debugging')
    (opts, args) = parser.parse_args()

    if len(args)< 1:
        raise SystemExit(parser.print_usage())
    pointing_models = args

## Get fake sky coverage points
    if opts.debug:
        skypoints = np.array(selectedcoverage())
        print 'Sky points (Az,El)'
        import pprint
        pprint.pprint(skypoints)
        testmodel = '-0:05:30.6, -0:00:03.3, 00:02:14, 00:00:02, -0:01:30.6, 00:08:42'
        print 'Test values: %s'%testmodel
        expectedvals = '-0.09183333333, -0.0009166666667, 0.03722222222, 0.0005555555556, -0.02516666667, 0.145'
        print 'Expected values: %s'%expectedvals
        testcoeffs = ['%.6f'%DDMMSS2DEG(np.array(coeff.split(':'),dtype=float)) for coeff in testmodel.strip().split(',')]
        print 'Calculated values: %s'% ','.join(testcoeffs)
        [p1, p3, p4, p5, p6, p7] = np.array(testcoeffs,dtype=float)
        beamrotation=getbeamrotation(skypoints[:,0], skypoints[:,1], p3, p4, p5, p6)
        print 'Az\tEl\tdX'
        for i in range(skypoints.shape[0]):
            print '%d\t%d\t%.6f' % (skypoints[i,0], skypoints[i,1], beamrotation[i])
        print beamrotation.reshape(3,4).T
        import sys
        sys.exit(0)

    elif opts.npoints is not None:
        # there are some degeneracy between terms in the equation, which may
        # cause this option to show excessive values
        skypoints = np.array(fakeskycoverage(opts.npoints))
    elif opts.full:
        skypoints = np.array(fullskycoverage())
    else: skypoints = np.array(selectedcoverage())

    fig, axes = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    axes.plot(skypoints[:,0], skypoints[:,1], 'y*', alpha=0.5)
    plt.title('Simulated sky coverage (Az,El)')
    plt.savefig('simulatedsky.png')

    models={}
    for model_file in pointing_models:
        ant_name = (os.path.basename(model_file).split('.')[0])
        if opts.ants is not None:
            if ant_name not in opts.ants: continue
        if not models.has_key(ant_name): models[ant_name]={}
        [model, coefficients] = readfile(model_file)
        try:
            [p1, p2, p3, p4, p5, p6, p7, p8] = coefficients
        except: # some models have less parameters
            [p1, p2, p3, p4, p5, p6, p7] = coefficients
        models[ant_name]['pointingmodel']=model
        models[ant_name]['beamrotation']=getbeamrotation(skypoints[:,0], skypoints[:,1], p3, p4, p5, p6)
        models[ant_name]['maxbeamrotation']=np.max(np.abs(models[ant_name]['beamrotation']))
    antennas=np.sort(models.keys())

    import operator
    plt.figure(figsize=(15,15))
    for i in range(len(antennas)):
        plotpoints = np.column_stack((skypoints, models[antennas[i]]['beamrotation']))
        plotpoints = np.array(sorted(plotpoints, key=operator.itemgetter(1,0)))
        ax = plt.subplot(4,1,i%4+1)
        plt.plot(plotpoints[:,1], plotpoints[:,2],'.--', label='%s'%antennas[i])
        box = ax.get_position()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Beamrotation')
    plt.xlabel('(Az,El) increasing in El for multiple Az')
    plt.savefig('beamrotation.png')

    maxbeamrotations = [models[ant]['maxbeamrotation'] for ant in antennas]
    diffmaxbeamrotations = maxbeamrotations-min(maxbeamrotations)
    outlier_idx = []
    outlier_idx = np.nonzero(diffmaxbeamrotations>3)[0]
    nantennas = np.arange(len(antennas))
    plt.figure(figsize=(8,8))
    plt.plot(nantennas, diffmaxbeamrotations,'go')
    # from Jira https://skaafrica.atlassian.net/browse/MKAIV-88
    spec = 3 # degrees
    plt.axhline(y=3, color='k', linestyle='--')
    if len(outlier_idx)>0:plt.plot(nantennas[outlier_idx], diffmaxbeamrotations[outlier_idx],'ro')
    plt.xticks(range(len(antennas)), antennas, size='small', rotation='vertical')
    plt.xlabel('Antennas')
    plt.ylabel('Differential Beam Rotation')
    plt.savefig('diffangle.png')

    if opts.verbose:
        from scipy.interpolate import griddata
        import matplotlib
        plt.figure(figsize=(7,30))
        for j in range(skypoints.shape[1]-2):
            plt.subplot(skypoints.shape[1]-2,1,j+1)
            grid = np.zeros((360,90))
            for i in range(skypoints.shape[0]):
                grid[skypoints[i,0],skypoints[i,1]]=np.abs(skypoints[i,j+2])
            grid_x, grid_y= np.mgrid[:grid.shape[0], :grid.shape[1]]
            points = grid.nonzero()
            values = grid[points]
            outgrid = griddata(points, values, (grid_x, grid_y), method='nearest')
            cmap = plt.cm.Spectral
            colormap=cmap(np.linspace(0,1,10))
            colormap_r = matplotlib.colors.ListedColormap(colormap)
            plt.imshow(outgrid)
            plt.colorbar()
            plt.axis('tight')
            plt.ylabel('Azimuth')
        plt.xlabel('Elevation')

##Summary
    diffparangle=diffmaxbeamrotations
    print
    print ' \tP1\t\tP3\t\tP4\t\tP5\t\tP6\t\tP7'
    print ' \tAz\t\tAz,El\t\tRF axis\t\tVertical\tVertical\tEl\t\tBeam rotation'
    print ' \toffset\t\torthogonality\torthogonality\ttilt N\t\ttilt E\t\toffset'
    for ant in antennas:
        coeffs = models[ant]['pointingmodel'].split(',')
        dX = models[ant]['maxbeamrotation']
        diffparangle = np.delete(diffparangle,[0], axis=0)
        print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6f' % (ant, coeffs[0].strip(), coeffs[2].strip(), coeffs[3].strip(), coeffs[4].strip(), coeffs[5].strip(), coeffs[6].strip(),dX)


    plt.show()

# -fin-
