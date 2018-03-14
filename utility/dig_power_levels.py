#!/usr/bin/python

import requests
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import optparse
from  datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

url = "http://portal.mkat.karoo.kat.ac.za/katstore/samples" # site for the most resent values
def get_ants(sensor, url, time_int, timest = None):
    """Get the number of antennas in MeerKAT
       900 seconds ago by querying the katstore
    url using each antenna CAM sensor.
    """
    
    if timest is None:
        timest=time.time()

        #Set limit to 1 for antenna.
        sensor_info = {
                      'sensor': sensor,
                      'start': timest -float(time_int),
                      'end':  timest,
                      'limit':1
                      }

    res = requests.get(url, params=sensor_info)
    katsi = re.compile('m\d{3}')
    ants = re.findall(katsi, res.content)
    return ants

def http_sensor_fetch(ant, key, time_int, timest = None):
     """ Fetch digitiser sensor values for each antenna from katstore.
     """
     if timest is None:
        timest=time.time()
        res = requests.get(url, params = {'sensor': ant+'_'+key,
                           'start': timest - float(time_int),
                           'end':  timest
                           }
                            )
    
        data_temp_list = [str(r[3]) for r in res.json()]
        return data_temp_list

def get_sensor_values(ants):
    """ Store digitiser sensor values per antenna into a structured array.
    """
    
    store_array = np.zeros((len(ants),1),
                          dtype = ([('antennas','S5'),
                          ('dig_selected_band','S5'),
                          ('rsc_rxl_serial_number','S5'),
                          ('dig_l_band_rfcu_hpol_rf_power_in', float),
                          ('dig_l_band_rfcu_vpol_rf_power_in',float),
                          ('dig_l_band_adc_hpol_rf_power_in',float),
                          ('dig_l_band_adc_vpol_rf_power_in',float),
                          ('dig_l_band_rfcu_hpol_attenuation',float),
                          ('dig_l_band_rfcu_vpol_attenuation',float)])
                          )
    
    #split sensor names into numeric and discrete for averaging. 
    numeric_sensors = [
             'dig_l_band_rfcu_hpol_rf_power_in',
             'dig_l_band_rfcu_vpol_rf_power_in',
             'dig_l_band_adc_hpol_rf_power_in',
             'dig_l_band_adc_vpol_rf_power_in',
             'dig_l_band_rfcu_hpol_attenuation',
             'dig_l_band_rfcu_vpol_attenuation'
               ]
    
    discrete_sensors=['dig_selected_band', 'rsc_rxl_serial_number']
     
    for i, ant in enumerate(ants):
        store_array[i]['antennas'] = ant
        
        #storing average of numeric sensor data sampled within 900 seconds into structured array 
        for key in numeric_sensors:
            Data_temp_list=http_sensor_fetch(ant,key, opts.time_interval)
            store_array[i][key] = np.mean([float(j) for j in Data_temp_list])
            
        #storing average of discrete sensor data sampled within 900 seconds into structured array     
        for key in discrete_sensors:
            Data_temp_list=http_sensor_fetch(ant, key, opts.time_interval)
            #print ant, Data_temp_list
            store_array[i][key] = Data_temp_list[0]
            
        print "Processing sensor information for antenna %s" % ant

    return store_array

def rfcu_calc(rfcuin_pol, rfcuout_pol, atten):
    """rcuinPol is a list of power in to digitser to the ADC per antenna and polarization (rfcu.pol.rf.power.in)
    rfcuoutPol is a list of power out of the ADC per antenna and polarization  (adc.pol.rf.power.in)
    atten is a list of attenuation levels per antenna and polarization.
    """
    #input pwr + 14dB (or 16dB) - atten = (-40+16-6)
    rfcuCalc=np.array(rfcuin_pol)+16-atten
    amp_gain=abs(np.array(rfcuin_pol))-(abs(np.array(rfcuout_pol))-atten)
    return  rfcuCalc,amp_gain

#target_rfcuin=-40dB
#target_rfcuout_min=-30dB
#target_rfcout_max=-3dB
def plot_rfcuPol(ampGain_pol, rfcuin_pol, rfcuout_pol, rfcuCalc_pol):
    """AmpGainPol is an array of Amplification gain estimated by subtracting the power levels reported by the adc.pol.rf.power.in
    and the rfcu.pol.rf.power.in
    rfcuCalcPol is an array of Calculated adc-in using the equation input pwr + 14dB (or 16dB) - atten.
    """
    fig=plt.figure(figsize=(20,12))
    ax1=plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(ampGain_pol,'mo-', lw=2,label='Amp gain')
    ax1.set_ylabel('Amplification Gain(dB)', fontsize=20)
    ax1.axhline(y=10, color='y', lw=1.5, linestyle='--', label='Min amp gain')
    ax1.axhline(y=16, color='orange', lw=1.5, linestyle='--', label='Max amp gain')
    plt.yticks(size=15)
    ax1.grid(linestyle='dotted')
    ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(1.0, 1.),
          fancybox=True, shadow=True)
    ax2=plt.subplot(212, sharex=ax1)
    ax2.set_ylabel('Power(dB)', fontsize=20)
    ax2.plot(rfcuin_pol,'bo-', lw=2,label='rfcu-in')
    ax2.plot(rfcuout_pol,'ro-', lw=2,label='system adc-in')
    ax2.plot(rfcuCalc_pol,'co-', lw=2,label='rfcu-in+16dB-atten')
    ax2.axhline(y=-40, color='b', lw=1.5, linestyle='--',label='Design rfcu-in')
    ax2.axhline(y=-30, color='r', lw=1.5, linestyle='--',label='Design Min adc-in')
    ax2.axhline(y=-3, color='r',lw=1.5, linestyle='-',label='Max adc-in')
    ax2.legend(fontsize=15, loc='upper center', bbox_to_anchor=(1.0, 1.8),
          fancybox=True, shadow=True)
    ax2.grid(linestyle='dotted')
    plt.xticks(range(len(antennas)), antennas, rotation=60, size=15)
    plt.yticks(size=15)
    ax2.set_xlabel('Antenna', fontsize=20)
    plt.subplots_adjust(hspace=.1)
    return fig

#Set up standard script options
parser = optparse.OptionParser(usage = '%prog [options]',
                               description = 'Check the health status of the digitiser and write an on pdf')
parser.add_option("-t", "--time-interval", default = 900.0,
                  help="time interval for querying katstore")                               
(opts,args) = parser.parse_args()

#get a list of antennas in MeerKAT
all_ants = get_ants('katpool_ants',url,opts.time_interval)
maint_receptors = list(sorted(get_ants('katpool_resources_in_maintenance',url,opts.time_interval)))
receptors = list(sorted(all_ants))
active_receptors = sorted(set(receptors)-set(maint_receptors))

print('MeerKAT receptors: {}\n\n{}'.format(len(receptors), receptors))
print ('Receptors in Maintanance:{}\n\n{}'.format(len(maint_receptors), maint_receptors))
print ('Active receptors {}\n\n{}'.format(len(active_receptors), active_receptors))
#Call main funtion to store sensor values
data = get_sensor_values(active_receptors)

#Extracting H_pol rfcu-in and adc-in from data
rfcuin_h = [np.asscalar(i) for i in data['dig_l_band_rfcu_hpol_rf_power_in']]
rfcuout_h = [np.asscalar(i) for i in data['dig_l_band_adc_hpol_rf_power_in']]

#Extracting  V_pol rfcu-in and adc-in from data
rfcuin_v = [np.asscalar(i) for i in data['dig_l_band_rfcu_vpol_rf_power_in']]
rfcuout_v = [np.asscalar(i) for i in data['dig_l_band_adc_vpol_rf_power_in']]

#Extracting all the antennas And attenuation
antennas = [np.asscalar(i) for i in data['antennas']]
att_h = np.ravel(data['dig_l_band_rfcu_hpol_attenuation'])
att_v = np.ravel(data['dig_l_band_rfcu_vpol_attenuation'])

#Calculating rfcu out
rfcuCalc_h = rfcu_calc(rfcuin_h, rfcuout_h, att_h)[0]
rfcuCalc_v = rfcu_calc(rfcuin_v, rfcuout_v, att_v)[0]
#Calculate the amplification gain
ampGain_h = rfcu_calc(rfcuin_h, rfcuout_h, att_h)[1]
ampGain_v = rfcu_calc(rfcuin_v, rfcuout_v, att_v)[1]
now = datetime.now()
pp = PdfPages('Digitiser_power_levels'+'_'+now.strftime("%Y-%m-%d-%H:%M")+'.pdf')

#Calling plot functions h & v
fig = plot_rfcuPol(ampGain_h, rfcuin_h, rfcuout_h, rfcuCalc_h)
plt.title('Digitiser power level-hpol',fontsize = 20, fontweight = 'bold')
fig.savefig(pp,format = 'pdf')
plt.close(fig)

fig=plot_rfcuPol(ampGain_v, rfcuin_v, rfcuout_v, rfcuCalc_v)
plt.title('Digitiser power level-Vpol', fontsize = 20, fontweight='bold')
fig.savefig(pp,format = 'pdf')
plt.close(fig)
pp.close()
plt.close('all')
