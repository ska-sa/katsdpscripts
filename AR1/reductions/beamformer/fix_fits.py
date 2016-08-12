from astropy.io import fits
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Fix .sf FITS.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', dest='path', type=str, action='store', default='',
                    help='path to the .sf file')
parser.add_argument('--psrcat', dest='psrcat', type=str, action='store', default='',
                    help='psrcat command of the form: psrcat -c "name p0 dm RAJ DecJ" 1644-455')
args = parser.parse_args()


def correct_fits(file_path, psrcat_command):
    """
    Maciej's fix for broken MK fits
    :param file_path: string file_path + file_name
    :return:
    """
    # TODO: param to add {key: value}
    hdulist = fits.open(file_path, mode="update", memmap=True, save_backup=False)
    hduP = hdulist[0]
    hduS = hdulist[2]

    proc = subprocess.Popen([psrcat_command], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    split_out = out.split('\n')[4].split()

    hduP.header["TELESCOP"] = "MeerKAT"
    hduP.header["RA"] = split_out[9]
    hduP.header["DEC"] = split_out[12]
    hduP.header["STT_CRD1"] = split_out[9]
    hduP.header["STT_CRD2"] = split_out[12]
    hduP.header["STP_CRD1"] = split_out[9]
    hduP.header["STP_CRD2"] = split_out[12]
    hduP.header["TRK_MODE"] = "TRACK"
    hduP.header["OBS_MODE"] = "SEARCH"
    hduP.header["TCYCLE"] = 0
    hduP.header["ANT_X"] = 5109318.8410
    hduP.header["ANT_Y"] = 2006836.3673
    hduP.header["ANT_Z"] = -3238921.7749
    hduP.header["NRCVR"] = 0
    hduP.header["CAL_MODE"] = "OFF"
    hduP.header["CAL_FREQ"] = 0.
    hduP.header["CAL_DCYC"] = 0.
    hduP.header["CAL_PHS"] = 0.
    hduP.header["CAL_NPHS"] = 0
    hduP.header["CHAN_DM"] = 0.0
    hduP.header["DATE-OBS"] = "2016-06-03T09:46:16"
    hduP.header["DATE"] = "2016-06-3T09:46:16"

    hduS.header["NPOL"] = 1
    hduS.header["POL_TYPE"] = "AA+BB"
    hduS.header["NCHNOFFS"] = 0
    hdulist.writeto(file_path.split('.')[0]+'_fix.'+file_path.split('.')[1])

pulsar_path = args.path
psr_comm = args.psrcat
correct_fits(pulsar_path, psr_comm)
print "Done"
