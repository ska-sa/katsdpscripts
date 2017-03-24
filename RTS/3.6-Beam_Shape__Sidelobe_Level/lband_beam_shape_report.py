#Add new row to 'RTS lband beam shape register' google spreadsheet, populating columns A-I:
#https://docs.google.com/a/ska.ac.za/spreadsheets/d/1JI-RPBAyoEOsKYCqZPNuS5DXD8GPjBJH0A9WCxToBPI/edit?usp=sharing
#and fill in row number below. Then restart kernel and run all cells.
%pylab inline
from matplotlib.backends.backend_pdf import PdfPages
import socket
import katholog
import time
import os
import gspread
from oauth2client.client import SignedJwtAssertionCredentials


# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces an l-band holography data file to produce a beam shape and sidelobe level report(s) in pdf format.')
parser.add_option( "--emss-path", default='',
                  help="directory where to find emss patterns', default = '%default'")
(opts, args) = parser.parse_args()

if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')

def opengooglespreadsheet():
    credentials = SignedJwtAssertionCredentials("client@holography-spreadsheet.iam.gserviceaccount.com", "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDW1YAB8YO7725u\nsaKx9PZEUXHfhgNsgSz4mTbntZ5nQqtP+RQOVWhwFN67G2IkZ34yOUfxKSoiiCIm\nJCfTR3W+mZqRov9R4b02vkmBBCmnhExypm5qOqcCKc6RlubWCKVdcb9z/8HWJgYT\nsxgCcV0px+e9xlfrHk+hS4TrWpCgIgJfkHJ/oeATLxYC3a1N0WsuZitCrZg2Mkkq\nO65Q4BoYpuv5+nA/ho4r7P73QoRcqhGJhgmZ1ueVPK25kwqY3UpCSYYXqSZXifJx\nza9son87zunzb7PU6UViqswa/oEF0+fo/sU9njnGwTZYRKZMGRZi3DnapQgH4oEm\nROAOzfdHAgMBAAECggEAA81znpUvx7vNMJhkUNINKdhnM5Wjqd//c8cCecd1Lk63\nyYqXGEnat2EnMBij/BD44tqws0wPOt09/X7WVZ8GLoF5gupnqoNcXgOwOWBhfF2T\nHNRlT+wJ2Lf0uA77tv4gLy88PElinO+/W8nnuuNWys6HA4znMXov3qL/j0rjFHJ9\nWuNWDBtEkBt+or3FmL7EUEsagKOlHOZo6M5JqWw5qm6Mx7Wx+Yi1WW+KonZYKCXM\nn1OFaDXaCpyx5HrXig0poWNvIh7fwIUwNaz0tQqXbVkQa2echH0IHqLTBppSTVeU\nNqE/vMO8LG2X6m+ihFFszgzUVrbc1seZIQpxqTPEwQKBgQDsjwasSNVtSSsCsyTd\nWS51HEXIKNftYwIjj/YjMXGFa69EhNp3SNz7ANrM/XyzEdDDBjqEBXlCddsnGxXX\nL0xMwC1yw22jzTz2zOCXiP2oFO13Qc4UCux/tLVD/bfAOIJrdNbo/Q/Y9FP6sr+U\nx2mpX2PAgsfg0ayqVCnpx1494wKBgQDofWgF5Utq+3AMo355xcUntnK/2OtZfapz\ndV97TAlDwhsjbYQ8OSjwRHl2Y3zac+gkxlpd6iguuwZ9iEINQa/1ABv12tn+Qpb5\nPybUW9qV11LvDzJp6zC4gu/eT0vZl/AKFySwjXj8JsgUzHXZ6jcfclZm5+4GL+89\nFGgMbRReTQKBgQCAXNP4JMV7OdrW6jK00bG95ouPI2qX68O7XGDpk+jPxzEh8x1A\n4Q7YPQx9c4d4+8/WI8kY3oeAIse6np3pWEcE1rtSrO0Pl0zfdyjf0Xwi+sgokFKs\n2Yife4Vo0YImEgPjH1GGt9sjlOEFBn2i09poB9TvH4gqXFxfSLA9pOtklQKBgFai\n1O7NgYs+Y4TyMCFkx5GC9cP0K8/PeoNIC+rAbPtpC//pwctHabAPdEvfyxkE9E8v\n82Dn701quIJzEloqTk24WrMFeRK88dGz7N5Z1FzePrODMEA0OpWnhYeMeTF+4x5/\nfValgZ5FPW4yuwAXva7kRrpWV2bK2hYi0ps+0sZJAoGAN8iN9IBPSQLolLcIY4sn\nDP7rAe58FUQKQINtb/UZoAXjIJNvzpuRd0eEdO3IcHfGjhVcEbQ68z1KOfOqnbk6\n8tpsO8L/gYHTEP6PhyMj8OpI0rwdqvPJmc82Bb83omIG+KIDYqestWZKMw1+O/VE\nB8Z3NjxVtbTtuLWUacpAoWM=\n-----END PRIVATE KEY-----\n".encode(), ['https://spreadsheets.google.com/feeds'])
    gc = gspread.authorize(credentials)
    wks = gc.open_by_key("1JI-RPBAyoEOsKYCqZPNuS5DXD8GPjBJH0A9WCxToBPI").sheet1 #google RTS l band beam shape spreadsheet
    return wks

def geterrorbeam(thisbeam,modelbeam):
    gridsize=modelbeam.shape[0]
    powbeam=20.0*np.log10(np.abs(modelbeam/modelbeam[gridsize/2,gridsize/2])).reshape(-1)
    dbeam=(np.abs(thisbeam)**2-np.abs(modelbeam)**2).reshape(-1)
    valid12dB=np.nonzero(powbeam>=-12)[0]
    dbeam[np.nonzero(np.isnan(dbeam))[0]]=0.0
    errorbeam=dbeam.reshape([gridsize,gridsize])
    maxbeam=np.max(np.abs(errorbeam).reshape(-1)[valid12dB])
    return errorbeam,maxbeam

def makereport(dataset,beams,beamemss):
    reportfilename='lband_beam_report_%s_%s_%s.pdf'%(dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],os.path.splitext(os.path.basename(filename))[0])
    gridsize=beamemss[0].gridsize
    margin=beamemss[0].margin
    with PdfPages(os.path.splitext(os.path.basename(filename))[0]+reportfilename) as pdf:
        figure(1,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*len(freqs))
            beams[ifreq].plot('Gx','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            beams[ifreq].plot('Dx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            beams[ifreq].plot('Dy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            beams[ifreq].plot('Gy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('Katholog version: %s Processed: %s\n%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\n'%(katholog.__version__,time.ctime(),dataset.filename,dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        pdf.savefig()

        figure(2,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            tmpGx=beams[ifreq].Gx;beams[ifreq].Gx=beams[ifreq].mGx
            tmpGy=beams[ifreq].Gy;beams[ifreq].Gy=beams[ifreq].mGy
            tmpDx=beams[ifreq].Dx;beams[ifreq].Dx=beams[ifreq].mDx
            tmpDy=beams[ifreq].Dy;beams[ifreq].Dy=beams[ifreq].mDy
            subplot(len(freqs),4,1+ifreq*len(freqs))
            beams[ifreq].plot('Gx','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            beams[ifreq].plot('Dx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            beams[ifreq].plot('Dy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            beams[ifreq].plot('Gy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
            beams[ifreq].Gx=tmpGx
            beams[ifreq].Gy=tmpGy
            beams[ifreq].Dx=tmpDx
            beams[ifreq].Dy=tmpDy
        suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\n'%('Polynomial fit',dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        pdf.savefig()
    
        figure(3,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*len(freqs))
            beamemss[ifreq].plot('Gx','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            beamemss[ifreq].plot('Dx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            beamemss[ifreq].plot('Dy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            beamemss[ifreq].plot('Gy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('EMSS model\n%dMHz %dMHz %dMHz %dMHz\n'%(freqs[0],freqs[1],freqs[2],freqs[3]))
        pdf.savefig()
    
        figure(4,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].Gx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            xlim([-extent/2,extent/2])
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            if (ifreq==0):
                legend(['0','45','90','135'])
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            xlim([-extent/2,extent/2])
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].Dy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            xlim([-extent/2,extent/2])
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].Gy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            xlim([-extent/2,extent/2])
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%(dataset.filename,dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        pdf.savefig()

        figure(5,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].mGx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            if (ifreq==0):
                legend(['0','45','90','135'])
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].mDx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].mDy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beams[ifreq].mGy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%('Polynomial fit',dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        pdf.savefig()
    
        figure(6,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            if (ifreq==0):
                legend(['0','45','90','135'])
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beamemss[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beamemss[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*len(freqs))
            bm=20.*np.log10(np.abs(beamemss[ifreq].Gy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('EMSS model\n%dMHz %dMHz %dMHz %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%(freqs[0],freqs[1],freqs[2],freqs[3]))
        pdf.savefig()

        figure(7,figsize=(14,12))
        clf()
        ext=margin[0]
        dGxmax=[0 for f in freqs]
        dGymax=[0 for f in freqs]
        mdGxmax=[0 for f in freqs]
        mdGymax=[0 for f in freqs]
        dGxstdev=[0 for f in freqs]
        dGystdev=[0 for f in freqs]
        for ifreq,thefreq in enumerate(freqs):
            dGx,dGxmax[ifreq]=geterrorbeam(beams[ifreq].Gx[0,:,:]/beams[ifreq].mGx[0,gridsize/2,gridsize/2],beamemss[ifreq].Gx[0,:,:]/beamemss[ifreq].Gx[0,gridsize/2,gridsize/2])
            dGy,dGymax[ifreq]=geterrorbeam(beams[ifreq].Gy[0,:,:]/beams[ifreq].mGy[0,gridsize/2,gridsize/2],beamemss[ifreq].Gy[0,:,:]/beamemss[ifreq].Gy[0,gridsize/2,gridsize/2])
            mdGx,mdGxmax[ifreq]=geterrorbeam(beams[ifreq].mGx[0,:,:]/beams[ifreq].mGx[0,gridsize/2,gridsize/2],beamemss[ifreq].mGx[0,:,:]/beamemss[ifreq].mGx[0,gridsize/2,gridsize/2])
            mdGy,mdGymax[ifreq]=geterrorbeam(beams[ifreq].mGy[0,:,:]/beams[ifreq].mGy[0,gridsize/2,gridsize/2],beamemss[ifreq].mGy[0,:,:]/beamemss[ifreq].mGy[0,gridsize/2,gridsize/2])
            mdGx=mdGx.reshape(-1)
            mdGy=mdGy.reshape(-1)
            dGx=dGx.reshape(-1)
            dGy=dGy.reshape(-1)
            dGxstdev[ifreq]=np.nanstd(mdGx-dGx)
            dGystdev[ifreq]=np.nanstd(mdGy-dGy)
            mdGx=np.abs(mdGx)
            mdGy=np.abs(mdGy)
            dGx=np.abs(dGx)
            dGy=np.abs(dGy)
            idx=np.nonzero(20.0*np.log10(np.abs(beamemss[ifreq].Gx[0,:,:].reshape(-1)))<-12)[0]
            idy=np.nonzero(20.0*np.log10(np.abs(beamemss[ifreq].Gy[0,:,:].reshape(-1)))<-12)[0]
            mdGx[idx]=np.nan
            dGx[idx]=np.nan
            mdGy[idy]=np.nan
            dGy[idy]=np.nan
            mdGx=mdGx.reshape([gridsize,gridsize])
            mdGy=mdGy.reshape([gridsize,gridsize])
            dGx=dGx.reshape([gridsize,gridsize])
            dGy=dGy.reshape([gridsize,gridsize])
        
            subplot(len(freqs),4,1+ifreq*len(freqs))
            imshow(mdGx,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx (max %.2f%%)'%(mdGxmax[ifreq]*100.0))
            subplot(len(freqs),4,2+ifreq*len(freqs))
            imshow(dGx,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gx (stddev %.2f%%)'%(dGxstdev[ifreq]*100.0))
            subplot(len(freqs),4,3+ifreq*len(freqs))
            imshow(mdGy,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy (max %.2f%%)'%(mdGymax[ifreq]*100.0))
            subplot(len(freqs),4,4+ifreq*len(freqs))
            imshow(dGy,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy (stddev %.2f%%)'%(dGystdev[ifreq]*100.0))
        suptitle('Error beam and polynomial fit within -12dB region\n%dMHz %dMHz %dMHz %dMHz\nMax error: %.2f%%'%(freqs[0],freqs[1],freqs[2],freqs[3],100.0*np.max([np.max(mdGxmax),np.max(mdGymax)])))
        pdf.savefig()

        d = pdf.infodict()
        d['Title'] = 'RTS L band beam shape report'
        d['Author'] = socket.gethostname()
        d['Subject'] = 'L band beam shape report'
        d['Keywords'] = 'rts holography'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()
    
def makedebugreport(dataset):
    debugreportfilename='lband_beam_debugreport_%s_%s_%s.pdf'%(dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]],os.path.splitext(os.path.basename(filename))[0])
    with PdfPages(os.path.splitext(os.path.basename(filename))[0]+debugreportfilename) as debugpdf:
        dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1,ignoreantennas=[])
        dataset.printenv()
        dataset.h5.select(dumps=np.array((np.array((dataset.ll)**2+(dataset.mm)**2<(dataset.radialscan_sampling)**2,dtype='int') & (1-dataset.flagmask)),dtype='bool'));
        corrprod_to_index = dict([(tuple(cp), ind) for cp, ind in zip(dataset.h5.corr_products, range(len(dataset.h5.corr_products)))])
        ifig=1
        for iant in range(len(dataset.radialscan_allantenna)):
            for jant in range(iant+1,len(dataset.radialscan_allantenna)):
                polprods = [("%s%s" % (dataset.radialscan_allantenna[iant],p[0].lower()), "%s%s" % (dataset.radialscan_allantenna[jant],p[1].lower())) for p in dataset.pols_to_use]
                cpindices=[corrprod_to_index.get(p) for p in polprods]

                figure(ifig,figsize=(16, 8), dpi=100)
                ifig+=1
                clf()
                subplot(4,1,1)
                imshow((np.angle(dataset.h5.vis[:,:,cpindices[0]].squeeze())))
                ylabel('time [sample]')
                title('XX')
                subplot(4,1,2)
                imshow((np.angle(dataset.h5.vis[:,:,cpindices[1]].squeeze())))
                ylabel('time [sample]')
                title('XY')
                subplot(4,1,3)
                imshow((np.angle(dataset.h5.vis[:,:,cpindices[2]].squeeze())))
                ylabel('time [sample]')
                title('YX')
                subplot(4,1,4)
                imshow((np.angle(dataset.h5.vis[:,:,cpindices[3]].squeeze())))
                ylabel('time [sample]')
                xlabel('channel')
                title('YY')
                suptitle(dataset.filename+": "+dataset.radialscan_allantenna[iant]+"-"+dataset.radialscan_allantenna[jant]+" "+dataset.target.name+"\nOn-target phase waterfall")
                debugpdf.savefig()

wks=opengooglespreadsheet()
usecycle='best'
filename=args[0]
ignoreantennas=ignoreantennastring.split(',') if (len(ignoreantennastring)>0) else []
extent=6
clipextent=6
bandwidth=4
freqs=[1100,1350,1500,1670]
#just determine scan_antennas
dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1)
availablescanantennas=[dataset.radialscan_allantenna[ant] for ant in dataset.scanantennas]

beamemss=[]
for thefreq in freqs:
    datasetemss=katholog.Dataset(opts.emss_path+'MK_GDSatcom_%d.mat'%thefreq,'meerkat',freq_MHz=thefreq,method='raw',clipextent=clipextent)
    datasetemss.visibilities=[np.conj(v) for v in datasetemss.visibilities]
    datasetemss.ll=-datasetemss.ll
    datasetemss.mm=-datasetemss.mm
    beamemss.append(katholog.BeamCube(datasetemss,interpmethod='scipy',xyzoffsets=[0,-13.5/2.0,0],extent=extent))
    try:
        beamemss[-1].fitpoly()
    except:
        beamemss[-1].mGx=beamemss[-1].Gx
        beamemss[-1].mDx=beamemss[-1].Dx
        beamemss[-1].mDy=beamemss[-1].Dy
        beamemss[-1].mGy=beamemss[-1].Gy

for scanantenna in availablescanantennas:
    ignoreantennas=[ant for in availablescanantennas].remove(scanantenna)
    if (usecycle=='best'):
        dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1,ignoreantennas=ignoreantennas)
        flags_hrs=dataset.findworstscanflags(freqMHz=freqs,dMHz=bandwidth,scanantennaname=dataset.radialscan_allantenna[dataset.scanantennas[0]],trackantennaname=dataset.radialscan_allantenna[dataset.trackantennas[0]],doplot=False)
        dataset.flagdata(flags_hrs=flags_hrs,ignoreantennas=ignoreantennas)
    elif (usecycle=='' or usecycle=='all'):
        print 'Using all cycles'
        dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1,ignoreantennas=ignoreantennas)
    else:
        dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1,ignoreantennas=ignoreantennas)
        cyclestart,cyclestop,nscanspercycle=dataset.findcycles(cycleoffset=0,doplot=False)
        cycleoffset=int((float(usecycle)-floor(float(usecycle)))*nscanspercycle)
        cyclestart,cyclestop,nscanspercycle=dataset.findcycles(cycleoffset=cycleoffset,doplot=False)
        cycles=zip(cyclestart,cyclestop)
        print 'Using cycle %d of %d with cycleoffset %d of %d'%(int(float(usecycle)),len(cycles),cycleoffset,nscanspercycle)
        cycle=cycles[int(float(usecycle))]
        dataset.flagdata(timestart_hrs=cycle[0],timeduration_hrs=cycle[1]-cycle[0],ignoreantennas=ignoreantennas)

    beams=[]
    for thefreq in freqs:
        beams.append(katholog.BeamCube(dataset,freqMHz=thefreq,dMHz=bandwidth,scanantennaname=dataset.radialscan_allantenna[dataset.scanantennas[0]],interpmethod='scipy',applypointing='Gx',extent=extent))
        try:
            beams[-1].fitpoly()
        except:
            beams[-1].mGx=beams[-1].Gx
            beams[-1].mDx=beams[-1].Dx
            beams[-1].mDy=beams[-1].Dy
            beams[-1].mGy=beams[-1].Gy
        
    makereport(dataset,beams,beamemss)
    wks=opengooglespreadsheet()
    wks.append_row([filename,'autoreduction',','.join(ignoreantennas),usecycle,'',dataset.target.name,dataset.radialscan_allantenna[dataset.scanantennas[0]],dataset.radialscan_allantenna[dataset.trackantennas[0]]])
    makedebugreport(dataset)