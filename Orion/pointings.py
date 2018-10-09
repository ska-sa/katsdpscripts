# (Re) create MeerKAT pointing list for the Orion cloud complex
# Do as 3 strips, skinny in RA, center strip centered on Orion KL
OKLRa = "05:35:14"
OKLDec = "-5:22:21"
#title = 'Orion Center'
#title = 'Orion Early'  # Before
title = 'Orion Late'   # After
# L band
import SkyGeom, ImageDesc
#MeerKAT = NVSS=26', scale to MeerKAT 26.*25./13.5 = 48'
delta = -16.*25./13.5 / 60   # Spacing in deg
#ftile  = 1; # first tile to do
#nTile = 1
#ftile  = 2; # early
#nTile = 2
ftile  = 3; # late
nTile = 3
nRow  = 10  # RA
nCol  = 30  # Dec
band="L"
freq = 1.3e9 # For MeerKAT
raJ = ImageDesc.PHMS2RA(OKLRa); decJ = ImageDesc.PDMS2Dec(OKLDec)
raTC = raJ+(nRow/2)*delta; decTC = decJ    # Top center ra,dec of first row of first tile
# >>> raJ,decJ = (83.808333333333323, -5.3724999999999996)

import SkyGeom,ImageDesc

# SkyGeom.PGal2Eq, SkyGeom.PBtoJ
# ImageDesc.PDMS2Dec, PDec2DMS
# ImageDesc.PHMS2RA, PRA2HMS
# T1R01C01;; Equatorial; J2000; +17:43:30.84737; -30:08:40.1788; ; ; ;

#fd = open("OrionCent","w")#
#fd = open("OrionEarly","w")
fd = open("OrionLate","w")
latarr=[]; lonarr=[]
odd = 0
for iTile in range(ftile,nTile+1):
    #toff = (iTile-1) * nRow * delta; # center, early
    toff = -(iTile-2) * nRow * delta; # late
    for iRow in range(0,nRow):
        xlong = raTC + toff - (iRow-1)*delta
        for iCol in range(1,nCol+1):
            #xlat = decTC + (iRow-nRow/2)*delta
            xlat = decTC + iRow*delta
            # Alternate rows and columns
            #if (odd%2)==0:
            if ((iRow%2==0) and (iCol%2==0)) or ((iRow%2!=0) and (iCol%2!=0)):
                xlat = decTC + (iCol-nCol/2)*delta
                latarr.append(xlat); lonarr.append(xlong)
                # to Strings
                rast  = ImageDesc.PRA2HMS(xlong).strip()
                decst = ImageDesc.PDec2DMS(xlat).strip()
                ras  = rast[0:2]+":"+rast[3:5]+":"+rast[7:]
                if decst[0:1]=='-':
                    decs = decst[0:3]+":"+decst[4:6]+":"+decst[7:]
                else:
                    decs = ' '+decst[0:2]+":"+decst[3:5]+":"+decst[6:]
                print ras,decs
                fd.write("('T%dR%2.2dC%2.2d', '%s', '%s'), \\\n"%(iTile, iRow, iCol, ras, decs))
                #print "T%dR%2.2dC%2.2d;; Equatorial; J2000; %s; %s; ; ; ;"%(iTile, iRow, iCol, ras, decs)
                #print ("%8.3f %8.3f")%(60*(toff + (iCol-1)*delta),60*((iRow-1-nRow/2)*delta)), iCol,iRow, ((iRow%2==0) and (iCol%2==0)) or ((iRow%2!=0) and (iCol%2!=0)) # Offsets
                #print "('T%dR%2.2dC%2.2d', '%s', '%s'), \"%(iTile, iRow, iCol, ras, decs)
            # end if want
            odd += 1  # next row use other
        #print iTile, iRow, iCol, ras, decs
        odd += 1
        # end col loop
# end loops
fd.close()

# plot
import OPlot, OErr
err = OErr.OErr()
radius = ((25./13.5)*45./(freq*1.0e-9))/120
#plot = OPlot.newOPlot("plot",err,output="Orion_Cent.ps/ps")
#plot = OPlot.newOPlot("plot",err,output="Orion_Early.ps/ps")
plot = OPlot.newOPlot("plot",err,output="Orion_Late.ps/ps")
info = plot.List
info.set("TITLE",title)
info.set("XLABEL","RA (deg)")
info.set("YLABEL","Dec (deg)")
#info.set("XMIN",min(lonarr)-radius)
#info.set("XMAX",max(lonarr)+radius)
#info.set("YMIN",min(latarr)-radius)
#info.set("YMAX",max(latarr)+radius)
info.set("JUST",1)
info.set("SSIZE",2)
OErr.printErr(err)
OPlot.PXYPlot (plot, 2, lonarr, latarr, err)
info.set("SSIZE",3)
#OPlot.PXYOver (plot, 12, lonarr, latarr, err)
for i in range (0,len(lonarr)):
    x=lonarr[i]; y=latarr[i]; 
    OPlot.PDrawCircle(plot,x,y,radius,err)
    OPlot.PDrawSymbol(plot,x,y,2,err)

OPlot.PShow(plot, err)




            
