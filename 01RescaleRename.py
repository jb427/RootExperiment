
#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#

#Load libraries:
import numpy as np
import os
import glob
import fnmatch
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Pool

#Input parameter:
##Folder name:
NamFold='../Data/RayAgarCP_1'
##Initial and last image number:
Ni=0
Nl=389
##Number of processor for post-processing parallelization:
NbProc=5

#Move in the working folder:
os.chdir(NamFold)

#Define global variables:
global row, col, timePic


#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#
#Rename file and create time vectors:
print('... is sorting pictures')

##Count the number of pictures in the folder to get the number of keeped steps:
NbStp=len(fnmatch.filter(os.listdir('.'), '*.jpg'))/2

##Initialise time vector:
timePic=np.zeros(NbStp)
##Create picture vector:
null=os.system('mkdir picturePl > /dev/null')
null=os.system('mkdir pictureWh > /dev/null')
##Loop over picture names:
cnt=0
for it1 in xrange(Ni,Nl+1):
    try:
        ###Find picture names:
        namCurPl=glob.glob('%04d'%it1+'_*_Pl.jpg')[0]
        namCurWh=glob.glob('%04d'%it1+'_*_Wh.jpg')[0]
        ###Extract time:
        timePic[cnt]=int(namCurPl[5:namCurPl[5:-1].index('_')+5])
        ###Rename pictures:
        null=os.system('mv '+namCurPl+' picturePl/'+'%04d'%cnt+'.jpg > /dev/null')
        null=os.system('mv '+namCurWh+' pictureWh/'+'%04d'%cnt+'.jpg > /dev/null')
        ###Increment:
        cnt+=1
    except:
        print('Picture '+'%04d'%it1+' is missing')

##Save time vector:
timePic-=np.amin(timePic)
np.savetxt('time.txt',timePic,delimiter='\n')

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#
#Rotate pictures:
print('... is rotating pictures')

##Function to rotate pictures: 
def RotatePicture(iPct):
    print iPct
    null=os.system('convert picturePl/'+'%04d'%iPct+'.jpg -rotate 90 picturePl/'+'%04d'%iPct+'.jpg > /dev/null')
    null=os.system('convert pictureWh/'+'%04d'%iPct+'.jpg -rotate 90 pictureWh/'+'%04d'%iPct+'.jpg > /dev/null')
    return 0

##Parallelization of the rotation:
p=Pool(NbProc)
null=p.map(RotatePicture,xrange(NbStp))

#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#
#Reframe pictures:

##Measure picture size:
pict1=misc.imread('picturePl/'+'%04d'%0+'.jpg')[:,:,1] #current
row,col=pict1.shape

##Function to measure the shift:
def MeasurePictureShift(iPct):
    global row, col
    print iPct
    ###Load pictures:
    pict1=misc.imread('picturePl/'+'%04d'%iPct+'.jpg')[:,:,1] 
    pict2=misc.imread('picturePl/'+'%04d'%(iPct+1)+'.jpg')[:,:,1]
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
    ###return the shifts:
    return xShift, yShift

##Parallelization of the shift computation:
print('... is centering pictures (measure displacement)')
p=Pool(NbProc)
xShiftVec0,yShiftVec0=zip(*p.map(MeasurePictureShift,xrange(NbStp-1)))

##Add the shifts:
xShiftVec=np.zeros(NbStp)
yShiftVec=np.zeros(NbStp)
for it in xrange(1,NbStp):
    xShiftVec[it]=xShiftVec[it-1]+xShiftVec0[it-1]
    yShiftVec[it]=yShiftVec[it-1]+yShiftVec0[it-1]

##Computation of the new image size:
MaxSft=np.amax(xShiftVec)+1
MinSft=np.amin(xShiftVec)-1
colN=int(col-MaxSft+MinSft)
rowN=row

##Rescale vectors:
xShiftVecN=(MaxSft-xShiftVec).astype(int)

##Function to reframe the pictures: 
def ReframePicture(iPct):
    print iPct
    ###Load pictures:
    pict0=misc.imread('picturePl/'+'%04d'%iPct+'.jpg')
    pict1=misc.imread('pictureWh/'+'%04d'%iPct+'.jpg') 
    ###Crop pictures: 
    pictF0=pict0[:,xShiftVecN[iPct]:xShiftVecN[iPct]+colN,:]
    pictF1=pict1[:,xShiftVecN[iPct]:xShiftVecN[iPct]+colN,:]
    ###Save picture:
    misc.imsave('picturePl/'+'%04d'%iPct+'.jpg',pictF0)
    misc.imsave('pictureWh/'+'%04d'%iPct+'.jpg',pictF1)

##Parallelization of the cropping:
print('... is centering pictures (crop)')
p=Pool(NbProc)
null=p.map(ReframePicture,xrange(NbStp))


##Select the area of interest:
###Open the first picture:
pict1=mpimg.imread('picturePl/'+'%04d'%0+'.jpg') 
###Select the left, right and down point:
plt.imshow(pict1)
plt.axis('off')
plt.title('clic left-right-down limits, and close') 
X=plt.ginput(3)
plt.show()
Imin0=int(X[2][1]); Jmin0=int(X[0][0]); Jmax0=int(X[1][0]);

##Function to crop the pictures: 
def CropPicture(iPct):
    print iPct
    ###Load pictures:
    pict0=misc.imread('picturePl/'+'%04d'%iPct+'.jpg')
    pict1=misc.imread('pictureWh/'+'%04d'%iPct+'.jpg') 
    ###Crop pictures: 
    pictF0=pict0[0:Imin0,Jmin0:Jmax0,:]
    pictF1=pict1[0:Imin0,Jmin0:Jmax0,:]
    ###Save picture:
    misc.imsave('picturePl/'+'%04d'%iPct+'.jpg',pictF0)
    misc.imsave('pictureWh/'+'%04d'%iPct+'.jpg',pictF1)

##Parallelization of the cropping:
print('... is cropping pictures')
p=Pool(NbProc)
null=p.map(CropPicture,xrange(NbStp))


#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#~/"\~#
#Create movies:

##Create picture vector:
null=os.system('mkdir tmpPl > /dev/null')
null=os.system('mkdir tmpWh > /dev/null')

##function to plot the pictures with the time: 
def PlotPicture(iPct):
    global timePic 
    print iPct
    ###Load the time:
    t0=timePic[iPct]
    h0,m0=divmod(t0,60)
    j0,h0=divmod(h0,24)
    ###Load pictures:
    pict0=mpimg.imread('picturePl/'+'%04d'%iPct+'.jpg')
    pict1=mpimg.imread('pictureWh/'+'%04d'%iPct+'.jpg') 
    ###Plot picture:
    plt.imshow(pict0)
    plt.axis('off')
    plt.title('%02d'%j0+'days   '+'%02d'%h0+'hours   ')
    plt.savefig('tmpPl/'+'%04d'%iPct+'.png',dpi=200)
    plt.close()
    plt.imshow(pict1)
    plt.axis('off')
    plt.title('%02d'%j0+'days   '+'%02d'%h0+'hours   ')
    plt.savefig('tmpWh/'+'%04d'%iPct+'.png',dpi=200)
    plt.close()

##Parallelization of the cropping:
print('... is plotting pictures')
p=Pool(2)
null=p.map(PlotPicture,xrange(NbStp))

##Make movies: 
print('... is making the movie')
null=os.system('ffmpeg -r 15 -f image2 -i tmpPl/%04d.png -qscale 1 Pl.avi > /dev/null')
null=os.system('ffmpeg -r 15 -f image2 -i tmpWh/%04d.png -qscale 1 Wh.avi > /dev/null')

##Remove folders:
null=os.system('rm -f -R tmpPl > /dev/null')
null=os.system('rm -f -R tmpWh > /dev/null')



