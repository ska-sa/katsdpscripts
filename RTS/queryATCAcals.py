#!/usr/bin/env python
# queryATCAcals.py  ,  last mod. 10/12/2015 [NJY]

#######################################################
# Use python web interface to query ATCA calibrator 
# database and find suitable sources for RTS/ MeerKAT.
#######################################################
import sys,os,time,json
import requests,urllib,urllib2,katpoint
import numpy as np
import matplotlib.pyplot as p
from optparse import OptionParser   # enables variables to be parsed from command

################################################
# Define functions to use to obtain valid data.
################################################
def get_array_res(file,freq,dishes):
    """Get valid VLA and ATCA configurations for given RTS baseline separation.
    Note: MeerKAT dish positions hard-coded from independent GPS -> ENU conversion by default. 
    """
    if file is None:
        ants = np.arange(64)
        ePos = np.float_('-8.32926 1.039484 -32.150089 -66.609204 -123.666093 -102.13475 -18.344533\
            -89.642824 -93.623242 32.267488 88.04202 83.917077 139.926412 236.702479 280.605117\
            210.621603 288.126491 199.566779 105.640585 170.739976 96.986093 -296.038955 -322.356972\
            -373.078365 -351.16367 -182.04871 -99.047837 40.454226 -51.214889 -88.838467 171.17841\
            246.467359 461.202988 580.66019 357.710994 386.13986 388.232105 380.239923 213.26078\
            253.722944 -26.885367 -287.593538 -361.779716 -629.965536 -896.222556 -1832.945369\
            -1467.454605 -578.361058 -2805.691765 -3606.049819 -2052.441373 -850.344829 -593.311534\
            9.317013 871.936958 1201.695027 1598.289246 294.557888 2805.693523 3686.290499\
            3419.584123 -16.507747 -1440.699688 -3419.582516'.split())
        nPos = np.float_('27.139624 62.605429 10.153473 32.101418 -18.463775 -48.667974 -61.019828\
            -168.295804 -300.597688 -136.642922 -277.470845 -117.691142 -133.808515 -159.038228\
            -51.304614 15.351851 48.54387 122.161946 -11.426972 -50.782268 -65.202743 -92.836115\
            92.239514 234.960394 384.481835 460.060248 251.512311 211.292254 147.269713 110.362197\
            348.436072 328.19189 409.934093 1098.416565 206.155841 53.520534 -56.299899 -224.872264\
            -334.678073 -357.764723 -477.788932 -427.242441 -225.845176 106.134998 834.890476\
            501.101685 1986.309359 -282.869399 2921.275843 670.857728 -609.221209 -534.975913\
            -914.22746 -1069.990281 -265.408557 330.972782 701.152358 3494.338657 2921.277682\
            993.374363 -1606.015868 -2089.343086 -2269.267591 -1606.01511'.split())
    else:
        ants,ePos,nPos,uPos = np.loadtxt(file,delimiter=',',unpack=True,dtype=float)
        ants = np.int_(ants)
    dishes = dishes.split()
    wLength = 2.998e8/freq
    
    # calculate beam FWHM for array config.
    if ( len(dishes) == 1 ):
        dSep = 13.5
        beamFWHM = (wLength/dSep)*(180./np.pi)
    elif( len(dishes) == 2 ):
        dishes = np.int_(dishes)
        eOff = np.abs(ePos[dishes[0]] - ePos[dishes[1]])
        nOff = np.abs(nPos[dishes[0]] - nPos[dishes[1]])
        dSep = np.sqrt(eOff**2 + nOff**2)
        beamFWHM = (wLength/dSep)*(180./np.pi)    
    elif ( len(dishes) > 2 ):
        print ' Enter valid number of dishes. Exiting now...'
        sys.exit(1)

    # determine which VLA and ATCA configs fit MeerKAT config
    VLAconfigs = [36e3,11e3,3.4e3,1.0e3]
    ATCAconfigs = [6e3,1.5e3]#,750,375]
    VLAres = np.array([(wLength/sep)*(180./np.pi) for sep in VLAconfigs])
    ATCAres = np.array([(wLength/sep)*(180./np.pi) for sep in ATCAconfigs])
    VLAindices = np.where(VLAres<=beamFWHM)[0]
    ATCAindices = np.where(ATCAres<=beamFWHM)[0]
    return dSep,beamFWHM,VLAindices,ATCAindices

def check_vla_res(bandIndex,inputArray,cIndices):
    """Check VLA source resolution."""
    discard = 'D0'
    if ( len(bandIndex) > 0 ):
        flux = inputArray[bandIndex,-1]
        cRes = inputArray[bandIndex,cIndices+2]
        indices = np.where(cRes!='?')
        res = cRes[indices][-1]
        if ( res == 'W' ) or ( res == 'X' ):
            discard = 'D1'
        vlaResInfo = '  '.join(inputArray[bandIndex][0].tolist()) # convert to string
        return [flux,vlaResInfo],discard
    else:
        return None,discard

def get_vla_info(source,band,cIndices):
    """Obtain VLA source information for source via ATCA calibrator database."""
    headers['Referer'] = 'http://www.narrabri.atnf.csiro.au/calibrators/calibrator_database_viewcal.html?source=' + str(source) + '&detailed=true'
    urlName = urllib2.quote(source.encode("utf8"))
    sourceInfo = None
    while sourceInfo is None:
        try:
            req = requests.post(url, data='action=info&source='+urlName, headers=headers)
            sourceInfo = req.json()
        except:
            pass
    ra = sourceInfo['rightascension']
    dec = sourceInfo['declination']
    rdStr = 'radec gaincal, ' + ra + ', ' + dec

    # get VLA data
    vla_data = sourceInfo['vla_text']
    if ( vla_data != '' ):
        vla_data = vla_data.split('\n')
        Jname = vla_data[0].split()[0]
        Bname = vla_data[1].split()[0]
        if ( Jname == '<a' ) or ( Jname == '<p' ):
            Jname = vla_data[0].split('>')[-1].split()[0] # reassign source name given html encoding
            Bname = vla_data[1].split()[0]
        vlaInfo = vla_data[5:]
        vlaInfArray = np.tile('     ',(len(vlaInfo),7))
        for index in np.arange(len(vlaInfo)):
            if ( len(vlaInfo[index].split()) < 7 ):
                vlaInfArray[index] = np.tile('None',7)
            else:
                vlaInfArray[index] = np.array(vlaInfo[index].split()[:7])
        if ( band == '4cm' ) or ( band == '15mm' ):
            index = np.where(vlaInfArray[:,1]=='X')[0]  # select 8.1 GHz band values
            result,discard = check_vla_res(index,vlaInfArray,cIndices)
            if result == None:
                index = np.where(vlaInfArray[:,1]=='U')[0] # select 15 GHz band values
                result,discard = check_vla_res(index,vlaInfArray,cIndices)
                if result == None:
                    result,discard = ['None','None'],'D0'
        elif ( band == '16cm' ):
            index = np.where(vlaInfArray[:,1]=='L')[0] # select 1.5 GHZ band values
            result,discard = check_vla_res(index,vlaInfArray,cIndices)
            if result == None:
                result,discard = ['None','None'],'D0'
        name = ' | '.join([Bname,Jname])
    else:
        name = source
        result = ['None','None']
        discard = 'D0'
    return [name,rdStr,result[1],discard]

def get_atca_fcoeffs(source,freq,band,bandLimits):
    """Obtain flux density coefficients for source in ATCA calibator database."""
    headers['Referer'] = 'http://www.narrabri.atnf.csiro.au/calibrators/calibrator_database_viewcal.html?source=' + str(source) + '&detailed=true'
    urlName = urllib2.quote(source.encode("utf8"))
    fluxInfo = None
    while fluxInfo is None:
        try:
            req = requests.post(url,data='action=band_fluxdensity&source='+urlName+'&band='+band,headers=headers)
            fluxInfo = req.json()
        except:
            pass

    # convert to katpoint units (i.e. MHz)
    coeffs = np.float_(fluxInfo['fluxdensity_coefficients'][:-1]) # ATCA flux coeffs
    M = np.array([[1, -3, 9], [0, 1, -6], [0, 0, 1]], dtype=float) # conversion matrix
#    M = np.array([[1,-3,9,-27], [0,1,-6,9], [0,0,1,-3],[0,0,0,1]], dtype=float) # conversion matrix
    padded_coeffs = np.zeros(3)
    padded_coeffs[:len(coeffs[:3])] = coeffs[:3]
    katCoeffs = np.dot(M,padded_coeffs)
    
    # obtain flux for given frequency and return valid params
    model = katpoint.FluxDensityModel(bandLimits[0],bandLimits[1],katCoeffs)
    flux = np.around(model.flux_density(freq/1e6),4)
    fluxErr = np.around(np.float_(fluxInfo['fluxdensity_scatter']),4)
    coeffStr = str(bandLimits[0])+ ' ' +str(bandLimits[1])+ ' '\
        +' '.join(np.array(katCoeffs,dtype=str).tolist())
    return np.float_(fluxInfo['observation_mjd']),flux,'('+coeffStr+')',fluxErr
    
def check_atca_res(source,band,dLim,cLim,cIndices):
    """Get amplitude phase closure and flux density defect estimates for source in ATCA database.
    Shortest ATCA baseline values matching RTS configuration returned.
    """
    headers['Referer'] = 'http://www.narrabri.atnf.csiro.au/calibrators/calibrator_database_viewcal.html?source='    + str(source) + '&detailed=true'
    urlName = urllib2.quote(source.encode("utf8"))
    quality = None
    while quality is None:
        try:
            req = requests.post(url,data='action=band_quality&source='+urlName+'&band='+band,headers=headers)
            quality = req.json()
        except:
            pass

    # test flux density defects & closure phases
    discard = 'D0'
    dArray = []
    cArray = []
    flags = ['6km','1.5km']#,'750m','375m']
    for flag in flags:
        try:
            dVal = float(quality[flag]['defect'])
        except:
            dVal = 0
        try:
            cVal = float(quality[flag]['closure_phase'])
        except:
            cVal = 0
        dArray = np.append(dArray,dVal)
        cArray = np.append(cArray,cVal)
    dArray = dArray[cIndices]
    cArray = cArray[cIndices]
    if ( np.size(dArray) > 0 ):
        for d in dArray:
            if d > dLim:
                discard = 'D1'
        for c in cArray:
            if c > cLim:
                discard = 'D1'
        return [np.around(dArray[-1],3),np.around(cArray[-1],3),discard]
    else:
        return [0,0,discard]

##########################
# Initialise parameters.
##########################
parser = OptionParser()
parser.formatter.max_help_position = 50
parser.formatter.width = 200
parser.add_option("-b", "--band", type="string", default='4cm',
    help="ATCA observing band override; choose from '16cm', '4cm' and '15mm' (default = '4cm').")
parser.add_option("--drange", type="string", default='-90 25',
    help="Declination range to search for calibrators (default = '-90 25').")
parser.add_option("--dishes", type="string", default='62 63',
    help="MeerKAT dishes to be used; 1 or 2 only (default = '62 63').")
parser.add_option("-f", "--freq", type="float", default=12.5e9,
    help="Sky frequency override (default = 12.5 GHz).")
parser.add_option("-i", "--infile", type="string", default=None, 
    help="File containing dish positions (default = None).")
parser.add_option("-n", "--names", type="string", default=None, 
    help="List of sources to obtain data for (default = None).")
parser.add_option("-o", "--outhdr", type="string", default='cals', 
    help="Output file header override (default = 'cals').")
parser.add_option("-s", "--fluxLim", type="float", default=2,
    help="Flux density limit override (default = 2 Jy).")
(opts,args) = parser.parse_args()
t0 = time.time() # record script start time

###################################################
# Set default search parameters for ATCA database.
###################################################
global values, payload, searchStr, url, headers
VLAsetup = np.array(['A','B','C','D'])
ATCAsetup = np.array(['6km','1.5km'])#,'750m','375m'])

values = {'rarange' : '0,24',
          'decrange' : ','.join(opts.drange.split()),
          'fluxlimit' : opts.fluxLim,
          'band' : opts.band}

payload = {'action' : 'search',
        'rarange': '0,24',
        'decrange' : ','.join(opts.drange.split()),
        'flux_limit' : opts.fluxLim,
        'flux_limit_band' : opts.band}

searchStr = urllib.urlencode(values).replace('%2C',',')
url = 'http://www.narrabri.atnf.csiro.au/cgi-bin/Calibrators/new/caldb_v3.pl' 
headers = {'Origin' : 'http://www.narrabri.atnf.csiro.au',
           'Cookie' : '_ga=GA1.2.92467305.1448360449; _referrer_og=https%3A%2F%2Fwww.google.co.za%2F;\
            _jsuid=413566380',
           'Accept-Encoding' : 'gzip, deflate',
           'Accept-Language' : 'en-US,en;q=0.8,de;q=0.6',
           'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)\
            Chrome/46.0.2490.86 Safari/537.36',
           'Content-Type' : 'application/x-www-form-urlencoded',
           'Accept' : '*/*',
           'Referer' : 'http://www.narrabri.atnf.csiro.au/calibrators/calibrator_database_search.html?'\
            + searchStr,
           'X-Requested-With' : 'XMLHttpRequest',
           'Connection' : 'keep-alive'}

if ( opts.band == '16cm' ):
    bandLims = [1100,3100]    # 1.1 - 3.1 GHz actual
elif ( opts.band == '4cm' ):
    bandLims=[7000,15000]     # 3.9 - 11 GHz actual
elif ( opts.band == '15mm' ):
    bandLims = [11000,25000]  # 16 - 25 GHz actual

#############################################################
# Obtain list of calibrators from port request to ATCA site.
##############################################################
if opts.names is None:
    data = None
    while data is None:
        try:
            req = requests.post(url, data=payload, headers=headers) # get cal list
            data = req.json()['matches']
            sources = [data[index]['name'] for index in np.arange(len(data))]
        except:
            pass
    print '\n Connecting to ATCA webserver to obtain suitable calibrators...' 
    print ' Number of sources with flux > %.2f Jy = %i @ %s'\
        %(values['fluxlimit'],len(sources),values['band'])
else:
    sources = np.array(opts.names.split())

###################################################
# Calculate beam FWHM for given baseline/dish
# and estimate suitable VLA & ATCA configurations.
####################################################
print '\n Calculating dish baseline and appropriate VLA/ATCA array configurations...'
dSep,beamFWHM,VLAindices,ATCAindices = get_array_res(None,opts.freq,opts.dishes)
print ' Dish baseline = %.3f m\n => Beam FWHM = %.3f arcmin @ %.3f GHz'\
     %(dSep,beamFWHM*60.,opts.freq/1e9)
print ' VLA matching configs:', VLAsetup[VLAindices]
print ' ATCA matching configs:', ATCAsetup[ATCAindices]

#################################################
# Successively run port requests for each source
# and store suitable calibrator info to file.
#################################################
print '\n Obtaining source information for all calibrators...'
flags = []
mjds = []
fluxVals = []
vStrings = []
fStrings = []
index = 0
for source in sources:
    print index, source
    sInfo = get_vla_info(source,opts.band,VLAindices)
    fInfo = get_atca_fcoeffs(source,opts.freq,opts.band,bandLims)
    qInfo = check_atca_res(source,opts.band,10,5,ATCAindices)
    flags = np.append(flags,qInfo[2])
    mjds = np.append(mjds,fInfo[0])
    fluxVals = np.append(fluxVals,fInfo[1])
    verboseStr = ', '.join(sInfo) + ', ' + fInfo[2] + ', ' +', '.join(np.array([fInfo[0],fInfo[1]],
        dtype=str))+', ' + ', '.join(np.array(qInfo,dtype=str))
    fmtedStr = ', '.join(sInfo[:2]) + ', ' + fInfo[2] + ', ' + str(fInfo[1])
    vStrings = np.append(vStrings,verboseStr)
    fStrings = np.append(fStrings,fmtedStr)
    index += 1

rIndices = np.where(flags == 'D0')[0]
fIndices = np.where(fluxVals>=opts.fluxLim)[0]
joinIndices = np.intersect1d(rIndices,fIndices)
sortIndices = np.argsort(fluxVals[joinIndices])[::-1] # descending order flux vals
np.savetxt(opts.outhdr+'.csv',fStrings[joinIndices][sortIndices],fmt='%s')
np.savetxt(opts.outhdr+'_verbose.txt',vStrings[joinIndices][sortIndices],fmt='%s')
#print fStrings[joinIndices][sortIndices]
#print mjds[joinIndices][sortIndices]
#print fluxVals[joinIndices][sortIndices]

#######################################
# Record and print total elapsed time.
#######################################
t1 = time.time()  
print '\n Total elapsed time: %.2f s (%.2f mins)\n' %(t1-t0,(t1-t0)/60.)
