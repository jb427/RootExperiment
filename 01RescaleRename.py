
#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Load libraries:
import numpy as np
import os
import glob
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Input parameter:
##Folder name:
NamFold='4'
##Initial and last image number:
Ni=0
Nl=709

#Move in the working folder:
os.chdir(NamFold)

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Rename file and create time vectors:
##Initialise time vector:
timePic=np.zeros(Nl-Ni+1)
##Create picture vector:
os.system('mkdir picturePl')
os.system('mkdir pictureWh')
##Loop over picture names:
for it1 in range(Ni,Nl+1):
    ###Find picture names:
    namCurPl=glob.glob('%04d'%it1+'_*_Pl.jpg')[0]
    namCurWh=glob.glob('%04d'%it1+'_*_Wh.jpg')[0]
    ###Extract time:
    timePic[it1]=int(namCurPl[5:namCurPl[5:-1].index('_')+5])
    ###Rename pictures:
    os.system('mv '+namCurPl+' picturePl/'+'%04d'%it1+'.jpg')
    os.system('mv '+namCurWh+' pictureWh/'+'%04d'%it1+'.jpg')

##Save time vector:
np.savetxt('time.txt',timePic,delimiter='\n')

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Rotate pictures:
for it1 in range(Ni,Nl+1):
    os.system('convert picturePl/'+'%04d'%it1+'.jpg -rotate 90 picturePl/'+'%04d'%it1+'.jpg')
    os.system('convert pictureWh/'+'%04d'%it1+'.jpg -rotate 90 pictureWh/'+'%04d'%it1+'.jpg')

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Reframe pictures:
##Initialisations:
xShiftVec=np.zeros(Nl-Ni)
yShiftVec=np.zeros(Nl-Ni)
xShift0=0
yShift0=0

pict1=misc.imread('picturePl/'+'%04d'%0+'.jpg')[:,:,1] #current
row,col=pict1.shape
##Loop over Pl picture to measure the shifts:
for it1 in range(Ni,Nl):
    print it1
    ###Load pictures:
    pict1=misc.imread('picturePl/'+'%04d'%it1+'.jpg')[:,:,1] 
    pict2=misc.imread('picturePl/'+'%04d'%(it1+1)+'.jpg')[:,:,1]
    ###Compute Fourier transform:
    pict1FFT=np.fft.fft2(pict1)
    pict2FFT=np.conjugate(np.fft.fft2(pict2))
    ###Convolute:
    pictCCor=np.real(np.fft.ifft2((pict1FFT*pict2FFT)))
    ###Compute the shift: 
    pictCCorShift=np.fft.fftshift(pictCCor)
    yShift,xShift=np.unravel_index(np.argmax(pictCCorShift),(row,col))
    yShift=yShift-int(row/2)
    xShift=xShift-int(col/2)
    ###Store the shift:
    xShift0=xShift0+xShift
    yShift0=yShift0+yShift
    xShiftVec[it1]=xShift0
    yShiftVec[it1]=yShift0

##Add initial values:
xShiftVec=np.insert(xShiftVec,0,0)
yShiftVec=np.insert(yShiftVec,0,0)

##Computation of the new image size:
MaxSft=np.amax(xShiftVec)+1
MinSft=np.amin(xShiftVec)-1
colN=int(col-MaxSft+MinSft)
rowN=row
##Rescale vectors:
xShiftVecN=(MaxSft-xShiftVec).astype(int)
##Loop over the pictures to reframe them: 
for it1 in range(Ni,Nl):
    print it1
    ###Load pictures:
    pict0=misc.imread('picturePl/'+'%04d'%it1+'.jpg')
    pict1=misc.imread('pictureWh/'+'%04d'%it1+'.jpg') 
    ###Crop pictures:
    pictF0=pict0[:,xShiftVecN[it1]:xShiftVecN[it1]+colN,:]
    pictF1=pict1[:,xShiftVecN[it1]:xShiftVecN[it1]+colN,:]
    ###Save picture:
    misc.imsave('picturePl/'+'%04d'%it1+'.jpg',pictF0)
    misc.imsave('pictureWh/'+'%04d'%it1+'.jpg',pictF1)
    

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Create movies:
##Create picture vector:
os.system('mkdir tmpPl')
os.system('mkdir tmpWh')

##Loop over the pictures to plot them: 
for it1 in range(Ni,Nl):
    print it1
    ###Load the time:
    t0=timePic[it1]
    h0,m0=divmod(t0,60)
    j0,h0=divmod(h0,24)
    ###Load pictures:
    pict0=mpimg.imread('picturePl/'+'%04d'%it1+'.jpg')
    pict1=mpimg.imread('pictureWh/'+'%04d'%it1+'.jpg') 
    ###Plot picture:
    plt.imshow(pict0)
    plt.axis('off')
    plt.title('%02d'%j0+'days   '+'%02d'%h0+'hours   ')
    plt.savefig('tmpPl/'+'%04d'%it1+'.png',dpi=250)
    plt.close()
    plt.imshow(pict1)
    plt.axis('off')
    plt.title('%02d'%j0+'days   '+'%02d'%h0+'hours   ')
    plt.savefig('tmpWh/'+'%04d'%it1+'.png',dpi=250)
    plt.close()

##Make movies: 
os.system('ffmpeg -r 15 -f image2 -i tmpPl/%04d.png -qscale 1 Pl.avi')
os.system('ffmpeg -r 15 -f image2 -i tmpWh/%04d.png -qscale 1 Wh.avi')

##Remove folders:
os.system('rm -f -R tmpPl')
os.system('rm -f -R tmpWh')




