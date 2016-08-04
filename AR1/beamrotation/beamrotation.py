from optparse import OptionParser

import os
import random

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

##Verbose example strings showing various usage options
def usagestr():
    usage = " \
              \npython %prog [options] <antenna_pointing_model_file(s)> \
              \nExamples: \
              \n\tCompare results of implementation with example xls sheet \
              \n\t> python %prog --debug \
              \n\tFor all antennas that have pointing models \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv \
              \n\tFor selected antennas \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' \
              \n\tSky coverage -- default is minimal \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' \
              \n\tSky coverage -- use specified az, el points (this option required both --az and --el if selected) \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' --az '0,90,180,270' --el '5,45,85' \
              \n\tSky coverage -- number of random sky points \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' -n 50 \
              \n\tSky coverage -- full az, el coverage \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' --full \
              \n\tSky coverage -- sampled az, el coverage distributed \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' --full --n 50 \
              \n\tSelected coefficients zeroed out \
              \n\t> python %prog katconfig/user/pointing-models/mkat/m0*.l.pm.csv --ants 'm001, m003, m010' --full --noskew --nobox --nonorthtilt --noeasttilt \
            "
    return usage

##Basic utilities
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
    data = fin.readline().strip()
    fin.close()
    return (data, [DDMMSS2DEG(np.array(coeff.split(':'),dtype=float)) for coeff in data.split(',')])

##Sky coverage simulation
def simulatecoverage(azlist, ellist):
    skypoints = []
    for elevation in ellist:
        for azimuth in azlist:
            skypoints.append([float(azimuth), float(elevation)])
    return skypoints
def minimalcoverage():
    return simulatecoverage([0, 90, 180, 270],[5,45,85])
def usercoverage(azlist, ellist):
    azlist = np.array(azlist.split(','),dtype=float)
    ellist = np.array(ellist.split(','),dtype=float)
    return simulatecoverage(azlist, ellist)
def fullskycoverage(delta_az=1., delta_el=1., minelev=15., maxelev=88.):
    ellist = np.arange(minelev, maxelev, float(delta_el))
    azlist = np.arange(0., 359., float(delta_az))
    return simulatecoverage(azlist, ellist)
def randomcoverage(nrpoints, minelev=15., maxelev=88.):
    skypoints = []
    for point in range(nrpoints):
        elevation=random.randrange(minelev,maxelev)
        azimuth=random.randrange(0.,359.)
        skypoints.append([float(azimuth), float(elevation)])
    return skypoints

##Beam rotation angle calculations
def APHbeamrotation(Az, El, p3, p4, p5, p6):
    return np.array((p4*np.sin(np.deg2rad(El))+p3+(p5*np.sin(np.deg2rad(Az))-p6*np.cos(np.deg2rad(Az))))*np.tan(np.deg2rad(El)))
def VLBImodelbeamrotation(Az, El, p3, p4, p5, p6):
    return np.array((p3 - p4/np.sin(np.deg2rad(El)) + p5*np.sin(np.deg2rad(Az)) - p6*np.cos(np.deg2rad(Az)))*np.tan(np.deg2rad(El)))

if __name__ == '__main__':

    parser = OptionParser(usage=usagestr(), version="%prog 1.0")
    parser.add_option('--ants', dest='ants', type=str, default=None,
                      help='List of antennas to evaluate')

    parser.add_option('--minimal', dest='min', action='store_true', default=False,
                      help='Minimal number points as sky coverage, default sky coverage')
    parser.add_option('--full', dest='full', action='store_true', default=False,
                      help='Get points over full sky coverage')
    parser.add_option('-n', '--npoints', dest='npoints', type=int, default=None,
                      help='Sky coverage simulated using npoint random samples of (Az,El) points')
    parser.add_option('--az', dest='az', type=str, default=None,
                      help='Sky coverage simulated using input list of azimuth angles')
    parser.add_option('--el', dest='el', type=str, default=None,
                      help='Sky coverage simulated using input list of elevation angles')

    parser.add_option('--vlbi', dest='vlbi', action='store_true', default=False,
                      help='Usage offset calculation from VLBI pointing model rather than XML calculation')
    parser.add_option('--noskew', dest='p3', action='store_true', default=False,
                      help='Ignore non-perpendicularity of az/el axes coefficient')
    parser.add_option('--nobox', dest='p4', action='store_true', default=False,
                      help='Ignore azimuth box offset coefficient')
    parser.add_option('--nonorthtilt', dest='p5', action='store_true', default=False,
                      help='Ignore azimuth ring tilt North offset')
    parser.add_option('--noeasttilt', dest='p6', action='store_true', default=False,
                      help='Ignore azimuth ring tilt East offset')

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False,
                      help='Show various additional graphs and output results')
    parser.add_option('--nopngs', dest='nopngs', action='store_false', default=True,
                      help='Do not save images as .png output files. Default is to create png images')
    parser.add_option('--debug',         dest='debug', action='store_true', default=False,
                      help='Show example output for debugging')
    (opts, args) = parser.parse_args()

    if len(args)< 1 and not opts.debug:
        raise SystemExit(parser.print_usage())
    pointing_models = args

    if bool(opts.az) != bool(opts.el):
        raise RuntimeError('When both --az and --el needs to be specified')

    if opts.ants is None:
        antennas = [os.path.basename(modelfile).split('.')[0] for modelfile in pointing_models]
    else:
        antennas = opts.ants.split(',')

## Debug output for implementation verification
    if opts.debug:
        skypoints = np.array(minimalcoverage())
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
        azoffsetangle=APHbeamrotation(skypoints[:,0], skypoints[:,1], p3, p4, p5, p6)
        print 'Az\tEl\tdX'
        for i in range(skypoints.shape[0]):
            print '%d\t%d\t%.6f' % (skypoints[i,0], skypoints[i,1], azoffsetangle[i])
        print azoffsetangle.reshape(3,4).T
        import sys
        sys.exit(0)

## Get fake sky coverage points
    if opts.az:
        skypoints = np.array(usercoverage(opts.az, opts.el))
    elif opts.npoints is not None and not opts.full:
        # there are some degeneracy between terms in the equation, which may
        # cause this option to show excessive values
        skypoints = np.array(randomcoverage(opts.npoints))
    elif opts.full:
        if opts.npoints is None:
            skypoints = np.array(fullskycoverage())
        else:
            delta_el = 75./float(opts.npoints)
            delta_az = 630./float(opts.npoints)
            skypoints = np.array(fullskycoverage(delta_az, delta_el))
    else:
        skypoints = np.array(minimalcoverage())

    fig, axes = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    axes.plot(np.deg2rad(skypoints[:,0]), skypoints[:,1], 'y*', alpha=0.5)
    plt.title('Simulated sky coverage (Az,El)')
    if opts.nopngs:
        plt.savefig('simulatedsky.png')

## Calculate azimuth beamrotation offset
    beamrotation={}
    for ant in antennas:
        ant = ant.strip()
        model_file = [ant_model for ant_model in pointing_models if ant in ant_model][0]
        [model, coefficients] = readfile(model_file)
        try:
            [p1, p2, p3, p4, p5, p6, p7, p8] = coefficients
        except: # some models have less parameters
            [p1, p2, p3, p4, p5, p6, p7] = coefficients
        if opts.p3:
            p3 = 0
        if opts.p4:
            p4 = 0
        if opts.p5:
            p5 = 0
        if opts.p6:
            p6 = 0

        if not beamrotation.has_key(ant):
            beamrotation[ant]={}
        beamrotation[ant]['pointingmodel']=model
        beamrotation[ant]['azimuth']=skypoints[:,0]
        beamrotation[ant]['elevation']=skypoints[:,1]
        if opts.vlbi:
            beamrotation[ant]['azoffsetangle']=VLBImodelbeamrotation(skypoints[:,0], skypoints[:,1], p3, p4, p5, p6)
        else:
            beamrotation[ant]['azoffsetangle']=APHbeamrotation(skypoints[:,0], skypoints[:,1], p3, p4, p5, p6)
        beamrotation[ant]['maxazoffsetangle']=np.max(np.abs(beamrotation[ant]['azoffsetangle']))

## Display calculated azimuth beam rotation
    antennas=np.sort(beamrotation.keys())[:-1] #ignore last antenna, dummy antenna

    import operator
    fig1 = plt.figure(figsize=(15,15))
    fig2 = plt.figure(figsize=(15,15))
    nants = 4
    if len(antennas) < 4:
        nants=len(antennas)
    for i in range(len(antennas)):
        ax1 = fig1.add_subplot(nants,1,i%nants+1)
        plotpoints = np.column_stack((beamrotation[antennas[i]]['azimuth'],
                                      beamrotation[antennas[i]]['elevation'],
                                      beamrotation[antennas[i]]['azoffsetangle']))
        plotpoints = np.array(sorted(plotpoints, key=operator.itemgetter(1,0)))
        ax1.plot(plotpoints[:,1], np.abs(plotpoints[:,2]),'.', label='%s'%antennas[i])
        box = ax1.get_position()
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_ylabel('Beamrotation')


        ax2 = fig2.add_subplot(1,1,1)
        elangle = np.unique(plotpoints[:,1])
        eloffset = []
        elerr = []
        for elev in elangle:
            elev_offset_range = np.abs(plotpoints[plotpoints[:,1]==elev,2])
            eloffset.append(np.median(elev_offset_range))
            elerr.append((np.max(elev_offset_range)-np.min(elev_offset_range))/2.)
        ax2 = plt.subplot(111)
        ax2.errorbar(elangle, eloffset, yerr=elerr, fmt='.:', label='%s'%antennas[i])
        box = ax2.get_position()
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_ylabel('Beamrotation')

    ax1 = fig1.add_subplot(nants,1,1)
    ax1.set_title('Calculated beamrotation offsets')
    ax1 = fig1.add_subplot(nants,1,nants)
    ax1.set_xlabel('(Az,El) increasing in El for multiple Az')

    ax2.set_title('Beamrotation offset trend (and uncertainty) over elevation')
    ax2.set_xlabel('Elevation')

    if opts.nopngs:
        fig1.savefig('beamrotationoffsets.png')
        fig2.savefig('beamrotationtrend.png')


## Display azimuth of max beam rotation error
    fig3 = plt.figure(figsize=(15,15))
    ax3 = fig3.add_subplot(1,1,1)
    for i in range(len(antennas)):
        plotpoints = np.column_stack((beamrotation[antennas[i]]['azimuth'],
                                      beamrotation[antennas[i]]['elevation'],
                                      beamrotation[antennas[i]]['azoffsetangle']))
        plotpoints = np.array(sorted(plotpoints, key=operator.itemgetter(1,0)))

        elangle = np.unique(plotpoints[:,1])
        maxoffset = []
        maxoffsetaz = []
        for elev in elangle:
            elev_offset_range = np.abs(plotpoints[plotpoints[:,1]==elev,2])
            azim_angles = plotpoints[plotpoints[:,1]==elev,0]
            maxoffset.append(elev_offset_range[np.argmax(elev_offset_range)])
            maxoffsetaz.append(azim_angles[np.argmax(elev_offset_range)])

        ax3.plot(elangle, maxoffsetaz, 'o:', label='%s'%antennas[i])
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_ylabel('Azimuth')
    ax3.set_xlabel('Elevation')
    ax3.set_title('Azimuth angle at which max beam rotation error is calculated')
    if opts.nopngs:
        fig3.savefig('maxbeamrotationazimuth.png')


    plt.figure(figsize=(11,15))
    nants = len(antennas)
    ncols = 3
    for i in range(len(antennas)):
        maxelevation = beamrotation[antennas[i]]['elevation'].max()
        maxelidx = [beamrotation[antennas[i]]['elevation']==maxelevation]
        x = np.abs(beamrotation[antennas[i]]['azoffsetangle'][maxelidx])
        # the histogram of the data
        ax = plt.subplot(int(nants/float(ncols))+1,ncols,i)
        n, bins, patches = plt.hist(x, facecolor='green', alpha=0.75)
        plt.title(antennas[i])
        plt.tight_layout()
    if opts.nopngs:
        plt.savefig('azoffsetdist.png')

## Display differential between maximum values
    maxazoffsetangles = [beamrotation[ant]['maxazoffsetangle'] for ant in antennas]
    diffmaxazoffsetangles = maxazoffsetangles-min(maxazoffsetangles)
    outlier_idx = []
    outlier_idx = np.nonzero(diffmaxazoffsetangles>3)[0]
    nantennas = np.arange(len(antennas))
    plt.figure(figsize=(8,8))
    plt.plot(nantennas, diffmaxazoffsetangles,'go')
    # from Jira https://skaafrica.atlassian.net/browse/MKAIV-88
    spec = 3 # degrees
    plt.axhline(y=3, color='k', linestyle='--')
    if len(outlier_idx)>0:plt.plot(nantennas[outlier_idx], diffmaxazoffsetangles[outlier_idx],'ro')
    plt.xticks(range(len(antennas)), antennas, size='small', rotation='vertical')
    plt.xlabel('Antennas')
    plt.ylabel('Differential Beam Rotation')
    if opts.nopngs:
        plt.savefig('diffangle.png')

##Summary
    # diffparangle=diffmaxazoffsetangles
    diffparangle=maxazoffsetangles
    print
    print ' \tP1\t\tP3\t\tP4\t\tP5\t\tP6\t\tP7'
    print ' \tAz\t\tAz,El\t\tRF axis\t\tVertical\tVertical\tEl\t\tBeam rotation'
    print ' \toffset\t\torthogonality\torthogonality\ttilt N\t\ttilt E\t\toffset'
    for ant in antennas:
        coeffs = beamrotation[ant]['pointingmodel'].split(',')
        dX = beamrotation[ant]['maxazoffsetangle']
        diffparangle = np.delete(diffparangle, [0], axis=0)
        print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6f' % (ant, coeffs[0].strip(), coeffs[2].strip(), coeffs[3].strip(), coeffs[4].strip(), coeffs[5].strip(), coeffs[6].strip(),dX)


    if opts.verbose:
        plt.show()

# -fin-
