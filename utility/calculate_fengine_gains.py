import numpy as np
import requests # Could use the builtin urllib but Requests are nicer.
import katdal 
import katpoint
import scikits.fitting as fit
from pynverse import inversefunc
import optparse
import matplotlib.pyplot as plt 

def get_power(data,name,pol):
    url="http://portal.mkat.karoo.kat.ac.za/katstore/samples" # this is for live data
    #url='http://kat-flap-cpt.mkat.control.kat.ac.za/katstore/samples' # flap in CT for older values
    start = katpoint.Timestamp(np.str(data.start_time))
    end = katpoint.Timestamp(np.str(data.end_time))
    #print start, end 
    params={'sensor': '%s_dig_l_band_adc_%spol_rf_power_in'%(name,pol),  # name of the sensor.
                   'start': start.secs, # start time in seconds - float should work.
                   'end': end.secs, # end time in seconds
                   'time_type':'s'} # Specify that we work in seconds. 
    res = requests.get(url, params) # Make the call.
    res.json() # Data comes back as JSON with the correct mime type set so Requests can convert for you.
    #data_dict = {r[0]: float(r[3]) for r in res.json()} # E.g. Making dict from the response. You'll prob do something different here.
    data_temp_list = [float(r[3]) for r in res.json()] # E.g. Making list from the response. You'll prob do something different here.
    return np.mean(data_temp_list),np.std(data_temp_list)

def get_power_array(data,name,pol):
    url="http://portal.mkat.karoo.kat.ac.za/katstore/samples" # this is for live data
    #url='http://kat-flap-cpt.mkat.control.kat.ac.za/katstore/samples' # flap in CT for older values
    start = katpoint.Timestamp(np.str(data.start_time))
    end = katpoint.Timestamp(np.str(data.end_time))
    #print start, end 
    params={'sensor': '%s_dig_l_band_adc_%spol_rf_power_in'%(name,pol),  # name of the sensor.
                   'start': start.secs, # start time in seconds - float should work.
                   'end': end.secs, # end time in seconds
                   'time_type':'s'} # Specify that we work in seconds. 
    res = requests.get(url, params) # Make the call.
    res.json() # Data comes back as JSON with the correct mime type set so Requests can convert for you.
    #data_dict = {r[0]: float(r[3]) for r in res.json()} # E.g. Making dict from the response. You'll prob do something different here.
    data_temp_list = [float(r[3]) for r in res.json()] # E.g. Making list from the response. You'll prob do something different here.
    return np.array(data_temp_list)
    
def get_gain_value(filename,no_ants,level=70,plot_graph=False,power=-30):
    #filename = '1536680347_sdp_l0.full.rdb'
    data = katdal.open(filename)
    ants = []
    for a in data.ants:
        if a.name not in no_ants :
            ants.append(a.name)
    data.select()
    nchan = data.channels.shape[0]
    mid = slice(np.int(2600./4096. * nchan ),np.int(3000./4096. * nchan) )
    p = {}
    data.select(corrprods='auto',scans='track',ants=ants)

    for j in range(data.shape[2]):
        if data.corr_products[j][0] == data.corr_products[j][1] :
            label=data.corr_products[j][0]
            pol = label[4]
            power,power_std = get_power(data,label[0:4],pol)
            p[label] = [power,power_std,None,[]]
    for scan in data.compscans() :
        #print scan[0],
        vis = data.vis[:,mid,:]
        #print scan
        for j in range(data.shape[2]):
            if data.corr_products[j][0] == data.corr_products[j][1] :
                label=data.corr_products[j][0]    
                gain_level = float(scan[1].split(',')[1])
                dat = np.median(np.abs(vis[:,:,j]),axis=[0,1])
                power,power_std,tmp,gain_arraylist = p[label]
                gain_arraylist.append((gain_level,dat))
                p[label] = [power,power_std,None,gain_arraylist]
                #print label,

    for keys in p :
        p[keys][-1] = np.array(p[keys][-1])

    level = level
    pvals = []
    for ant in np.sort(p.keys()) :
            #print ant
            poly = fit.PiecewisePolynomial1DFit()
            valid = (p[ant][3][:,1] > 10) & (p[ant][3][:,1] < 2000) 
            if valid.sum()> 0 and p[ant][3][valid,1].sum() > 0 :
                poly_func = poly.fit(p[ant][3][valid,0],p[ant][3][valid,1])
                try:
                    fits = inversefunc(poly_func, y_values=level,domain=(p[ant][3][valid,0].min(),p[ant][3][valid,0].max()))
                except :
                    print "Error inverting function ",ant
                    fits = 0.0
                p[ant][2] = fits
                pvals.append([p[ant][0],p[ant][2]*1])
                #if fits < lowlim or fits > hilim:
                #    print ant,fits
            else :
                print " Error  no valid values:",ant 
                pass
                
    pvals = np.array(pvals)
    lowlim , hilim = np.median(pvals[:,1])-np.std(pvals[:,1])*2,np.median(pvals[:,1])+np.std(pvals[:,1])
    meanv = np.median(pvals[:,1])
    mask = (pvals[:,1] > lowlim) *  (pvals[:,1] < hilim)
    linear = fit.LinearLeastSquaresFit()
    linear_func = linear.fit(pvals[mask,0],pvals[mask,1])
    print data.description,linear_func(-30)[0]
    if plot_graph :
        fig, ax = plt.subplots(figsize=(20,10))
        for ant in np.sort(p.keys()) :
            if p[ant][2] > 0 :
                plt.errorbar(p[ant][0],p[ant][2],xerr=p[ant][1],fmt='k.')
        #plot(poly_func(np.linspace(125,0,20)),np.linspace(125,0,20),'r')
        plt.ylabel("F-engine gain for a nominal correlator Level=%3.0f"%(level))
        plt.xlabel("ADC power level")
        a,b = plt.xlim()
        plt.xlim(-40,b)
        #ax.plot(range(20))
        ax.axvspan(-40, -37, alpha=0.5, color='red')
        ax.axvspan(-37, -31, alpha=0.5, color='yellow')
        ax.axvspan(-31, np.max([b,-5]), alpha=0.3, color='green')
        plt.title(data.description)
        plt.text(-38.5, meanv, 'Error', fontsize=20)
        plt.text(-35, meanv, 'Warning', fontsize=20)
        plt.grid()

    return linear_func(power)[0]#,p





# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <data file>",
                               description="This processes a dataset and extracts fitted gain parameters "
                                           "from the scans in it.")
parser.add_option( "--exclude-ants", dest="ex_ants",default=None,
                  help="List of antennas to exculde from the reduction "
                       "default is None of the antennas in the data set")
parser.add_option( "--plot",action="store_true",
                  default=False, help="Produce a graph of digitiser power vs. optimal gain level ")

parser.add_option("-l", "--level", default=70.0, help="optimal correlator power to fit for , default( %default)")
parser.add_option("--power", default=-30.0, help="ADC power to select for , default( %default)")

parser.add_option("-o", "--output", dest="outfilebase",default=None,
                  help="Base name of output file "
                       "default is '<dataset_name>_fengine_gain')")

(opts, args) = parser.parse_args()
if opts.ex_ants is None :
    ex_ants = []
else :
    ex_ants = opts.ex_ants
gain = get_gain_value(args[0],ex_ants,plot_graph=opts.plot,level=opts.level,power=opts.power)
print("Optimal F-engine requantion for a target level of %f for a ADC power of %f is %f"%(opts.level,opts.power,gain))