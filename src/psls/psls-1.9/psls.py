#!/usr/bin/env python
import numpy as np
import math
import sls
import os
import re 
import yaml
import sys
import getopt
import shutil
import transit as tr
from scipy.stats import chi2
from packaging.version import parse as parse_version
import spotintime
from scipy.signal import lombscargle
from astropy.timeseries import LombScargle
import flares
import h5py
import string
'''

PSLS : PLATO Solar-like Light-curve Simulator

If you use PSLS in your research work, please make a citation to Samadi et al (2019, A&A, 624, 117, https://www.aanda.org/articles/aa/abs/2019/04/aa34822-18/aa34822-18.html)
and Marchiori et al (2019, A&A, 627, A71, https://www.aanda.org/articles/aa/abs/2019/07/aa35269-19/aa35269-19.html)

Copyright (c) October 2017, R. Samadi (LESIA - Observatoire de Paris)

This is a free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this code.  If not, see <http://www.gnu.org/licenses/>.

'''


__version__ = 1.9

jupiterRadius = 71492.0  # km
ua2Km = 149.59e6  # km
# NSR values computed by V. Marchiori (see Marchiori et al 2019, A&A)
NSR_Pmag = np.array([7.76,8.16,8.66,9.16,9.66,10.16,10.66,11.16,11.66,12.16,12.56,12.76,13.16,13.66,14.16,14.66,15.16,15.56])# P magnitude
NSR_Vmag = NSR_Pmag + 0.34 # V magnitude
NSR_values_24 = np.array([10.6,12.9,16.4,20.8,26.7,34.5,44.8,59.2,79.1,106.8,138.5,156.8,205.8,298.6,442.9,668.8,1018.9,1444.4])
NSR_values = NSR_values_24*math.sqrt(24.)

def VmP(teff):
    '''
	Given the star Teff, return V-P according to Eq. 8 in  Marchiori et al (2019)
	'''
    # return -1.238e-12*teff**3 + 4.698e-8*teff**2 - 5.982e-4*teff + 2.506 # non-compliant with  Marchiori et al (2019)
    return -1.184e-12*teff**3 + 4.526e-8*teff**2 - 5.805e-4*teff + 2.449  # compliant  with  Marchiori et al (2019)

def generateZ(orbitalPeriodSecond, planetSemiMajorAxis,
                  starRadius, SamplingTime, IntegrationTime, TimeShift, sampleNumber,
                  orbitalStartAngleRad, p):
    '''
    :INPUTS:
    orbitalPeriodSecond = orbital period of the planet in second
    planetSemiMajorAxis = semi Major axis in km
    starRadius = star radius in km
    SamplingTime:int = Sampling time in second, (Plato = 25s)
    IntegrationTime : integration time in seconds
    TimeShift: time shift in seconds
    sampleNumber = number of sample we want (==> z.size)
    orbitalStartAngleRad = orbital angle in radians where to start planet position
    p = rp / r*
    :OUTPUTS:
    time: the time series
    z: z = d / r*, is the normalized separation of the centers (sequence of positional offset values)
    transit_dates: the date of each transit (in days)

    E. Grolleau
    L.C. Smith (for the vectorized version, 13.04.2021)
    '''
    angleIncrement = SamplingTime * 2.0 * math.pi / orbitalPeriodSecond
    angle0 = orbitalStartAngleRad + (IntegrationTime/2. + TimeShift)* 2.0 * math.pi / orbitalPeriodSecond
    angles = angleIncrement * np.arange(sampleNumber) +  angle0
    # For occultquad computation we need that z < p+1
    time = np.arange(sampleNumber) * SamplingTime + IntegrationTime/2. + TimeShift
    z = np.where(
        np.sin(angles)>0,
        np.abs((planetSemiMajorAxis / starRadius) * np.cos(angles)),
        p+1
    )
    z = z.clip(min=0, max=p+1)

    t0 = angle0*orbitalPeriodSecond/2./math.pi + orbitalPeriodSecond/4.
    tnb = int(math.floor((sampleNumber*SamplingTime-t0)/orbitalPeriodSecond)) + 1
    transit_dates =  t0 + np.arange(tnb) * orbitalPeriodSecond
    transit_dates /= 86400.


    phi0 = np.arccos((p+1)/np.abs((planetSemiMajorAxis / starRadius)))
    t0 = phi0*orbitalPeriodSecond/2./math.pi/86400.
    phi2 = np.arccos(-(p+1)/np.abs((planetSemiMajorAxis / starRadius)))
    t2 = phi2*orbitalPeriodSecond/2./math.pi/86400.
    t1 = (t0+t2)/2.
    transit_phases = [t0,t1,t2] # start, middle and end dates of the first transit
    return (time,z,transit_dates,transit_phases)


def psd (s,dt=1.):
    '''
 Inputs:
 s : signal (regularly sampled)
 dt: sampling (seconds)

 Outputs:
 a tuple (nu,psd)
 nu: frequencies (Hz)
 psd: power spectral density  (in Hz^-1). A double-sided PSD is assumed
    '''

    ft=np.fft.fft(s)
    n=len(s)
    ps=(np.abs(ft))**2*(dt/n)
    nu=np.fft.fftfreq(n,d=dt)
    nnu=n//2
    return (nu[0:nnu],ps[0:nnu])

def LS(t,s,dt):
    '''
    Compute a LombScargle periodogram
    
    t: time (sec)
    s: the signal
    dt: mean sampling time (sec)
    
    return:
    nu: frequencies (Hz)
    PSD: the PSD /Hz
    
    '''
    nt = len(s)
    duration = t[-1]-t[0]
    dnu = 1./duration # freq. resolution Hz
    nnu = nt//2
    nu = (np.arange(nnu)+1)*dnu
    psd = LombScargle(t, s, normalization='psd').power(nu)
    psd *=  dt
    return nu,psd


def AddFlare(time,FlareMeanPeriod,FlareUpDown,FlareAmplitude,FlareMeanDuration,FlareDurationDispersion,seed):

    FlareMeanPeriodSec = FlareMeanPeriod*86400. # days -> sec
    if(FlareMeanDuration<=0):
        FlareMeanDurationSec = FlareMeanPeriodSec/5.
    if(FlareDurationDispersion):
        FlareDurationDispersionSec = FlareMeanPeriodSec/20.

    tmin = time[0]
    tmax = time[-1]

    FlareNumbers = int((tmax - tmin) / FlareMeanPeriodSec)
    np.random.seed(seed)
    t_flares = np.random.uniform(low=tmin, high=tmax, size=FlareNumbers)
    amp_flares = np.random.normal(loc=FlareAmplitude, scale=FlareAmplitude / 10, size=FlareNumbers)
    amp_flares *= 1e-6 # ppm -> normalized unit
    duration_flares = np.random.normal(loc=FlareMeanDurationSec, scale=FlareDurationDispersionSec, size=FlareNumbers)
    LC = np.ones(time.size)
    return flares.add_flares(LC,time,t_flares,amp_flares,duration_flares,FlareUpDown)

def platotemplate(duration,dt=1.,V=11.,n=24,residual_only=False,cl=None):
    '''

 Return the total noise budget (in  ppm^2/ Hz) as a function of frequency.
 The budged includes all the random noise (including the  photon noise) and the resdiual error (after all corrections)
 It is assumed that the residual error is not correlated among the telescopes
 
 Inputs:
 duration : in days
 dt :sampling time i seconds
 V : star magnitude
 n : number of telescope (default: 24)
 
 Outputs:
 a tuple (nu,psd)
 nu: frequencies (Hz)
 psd: power spectral density  (in ppm^2 / Hz)

 cl: confidence level (<1), if specified the mean white noise level is multiplied by the threshold corresponding to the given confidence level, ,if not specified (None) the function returns the mean noise level
 '''
    V0 = 11 # reference magnitude
    scl = (24./n) # we assume that all the noises including the residual error are not correlated over the telescopes
    sclpn = 10.**( (V-V0)/2.5 ) # scaling applied on the random noise only
    if(cl!=None):
        threshold = chi2.ppf(cl,2)/chi2.mean(2) # 2 is the degree of freedom
        sclpn *= threshold
    n=int(np.ceil(86400.*duration/dt) )
    nu=np.fft.fftfreq(n,d=dt)
    m = int(n/2.)
    nu=nu[0:m]
    nu0=20e-6 # R-SCI-350, R-SCI-342 
    nu1=3e-6 #  R-SCI-350, R-SCI-342 
    s0 = 0.68 * 1e3 # R-SCI-342 
    s1 = 50.*1e3  #  R-SCI-350
    s3 = 3.0 * 1e3  # [ppm/Hz^(1/2)] random noise level at V=11 for 24 telescopes (equivalent to 50 ppm/hr)

    ps=np.zeros(m)
    j=np.where( nu >= nu0)
    if( j != -1):
        ps[j[0]] = (s0**2 + s3**2*sclpn*(residual_only==False))*scl
    j=np.where( (nu < nu0) & (nu>0.) )
    if( j != -1):
        ps[j[0]] = (np.exp( np.log(s1) + (np.log(s0)-np.log(s1)) * ((np.log(nu[j])-np.log(nu1))/(np.log(nu0)-np.log(nu1)) ))**2  + s3**2*sclpn*(residual_only==False))*scl
    return (nu,ps)
    

def pip(x,y,poly):
    '''
    test if a point is inside a polygon
    
    Taken from: http://geospatialpython.com/2011/01/point-in-polygon.html
    
    '''
    n = len(poly)
    inside = False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside


def rebin1d(array,n):
    nr=int(float(array.shape[0])/float(n))
    return (np.reshape(array,(n,nr))).sum(1)

# ModelDir= 'grid_v0.1_ov0-plato/'


def search_model_hdf5(ModelFile, ES, logg, teff, dlogg=0.01, dteff=15., verbose=False, plot=False):

    if(verbose): print('loading %s ' % (ModelFile))
    pack = h5py.File(ModelFile)
    nt = len(pack.keys()) # number of tracks
    if(verbose): print('number of tracks: %i' % nt)
    nm = 100000 # maximum number of steps per track
    teffG = -100.*np.ones((nt,nm))
    loggG = -100.*np.ones((nt,nm))
    massG = -100.*np.ones((nt,nm))
    radiusG = -100.*np.ones((nt,nm))
    Xc = -100.*np.ones((nt,nm))
    tracknames  = np.zeros(nt,dtype='U20')
    i = 0
    for key in pack.keys():
        if (key != 'license'):
            glob = pack[key]['global']
            ns = len(glob['teff'])
            teffG[i,:ns] = np.array(glob['teff'])
            loggG[i,:ns] = np.array(glob['logg'])
            massG[i,:ns] = np.array(glob['mass'])
            radiusG[i,:ns] = np.array(glob['radius'])
            Xc[i,:ns] = np.array(glob['Xc'])
            tracknames[i] = key
        i += 1

    if (ES.lower() == 'any'):
        sel = np.ones((nt,nm),dtype=bool)
    elif (ES.lower() == 'ms'):
        sel = (Xc > 1e-3)
    elif (ES.lower() == 'sg'):
        sel = (Xc <= 1e-3)  & (Xc > -100.)
    else:
        raise sls.SLSError("unmanaged evolutionary status:" + ES)

    if (sel.sum() == 0):
        raise sls.SLSError("no models full fill the criteria ")

    Chi2 = ((teffG - teff) / dteff) ** 2 + ((loggG - logg) / dlogg) ** 2

    Chi2[sel == False] = 1e99

    i = np.argmin(Chi2)
    j = i//nm
    k = i%nm
    teffb = teffG[j,k]
    loggb = loggG[j,k]
    massb = massG[j,k]/sls.msun
    radiusb = radiusG[j,k]/sls.rsun
    name =  ('%s/osc/%i' % (tracknames[j],k))
    modes = pack[name]

    if (verbose):
        print(('Best matching, teff = %f ,logg = %f, M = %f, R = %f, Chi2 = %f') % (teffb, loggb,massb,radiusb,Chi2[j,k]))
        print(('Star model name: %s') % (name))


    if (plot):
        plt.figure(200)
        plt.clf()
        sel = (teffG>-100) & (loggG > -100.)
        plt.plot(teffG[sel], loggG[sel], 'k+')
        plt.plot([teffb], [loggb], 'ro')
        plt.plot([teff], [logg], 'bo')
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.ylabel(r'$log g$')
        plt.xlabel(r'$T_{\rm eff}$ [K]')
        plt.draw()

    return modes, teffb, loggb,massb,radiusb

def search_model(ModelDir,ES,logg,teff,dlogg=0.01,dteff=15.,verbose=False,plot=False):

    pack = np.load(ModelDir+'data.npz')
    files = pack['files']
    glob = pack['glob'] # star global parameters
    ## references = pack['references'] # references (&constants) parameters
    teffG = glob[:,17]
    loggG = glob[:,18]
    numaxG = glob[:,28]
    massG = glob[:,0]
    radiusG = glob[:,1]
    if(verbose):
        print("Searching the model in the 'old' grid")
    if(ES.lower() == 'any'):
        sel = np.ones(glob.shape[0],dtype=bool)
    elif(ES.lower() == 'ms'):
        sel = (numaxG > 1e-3)
    elif(ES.lower() == 'sg'):
        sel = (numaxG <= 1e-3) & (numaxG < 200.)
    else:
        raise sls.SLSError("unmanaged evolutionary status:"+ ES)
    
    if(sel.sum()==0):
        raise sls.SLSError("no models full fill the criteria ")
    
    loggG = loggG[sel]
    teffG = teffG[sel]
    massG = massG[sel]
    radiusG = radiusG[sel]

    files = files[sel]
    Chi2 = ((teffG-teff)/dteff)**2 +  ((loggG-logg)/dlogg)**2
    
    i = np.argmin(Chi2)
    teffb = teffG[i]
    loggb = loggG[i]
    massb = massG[i]
    radiusb = radiusG[i]

    filename = files[i]
    if(type(filename) == bytes or type(filename) == np.bytes_): # solve a compatibility issue with strings coded as bytes
        filename = filename.decode()
    print(filename)
    name = re.sub('-nad.osc','',os.path.basename(filename))
    
    if(verbose):        
        print(('Best matching, teff = %f ,logg = %f, M = %f, R = %f, Chi2 = %f') % (teffb, loggb,massb,radiusb,Chi2[i]))
        print(('Star model name: %s') % (name))
        
    if(plot):
        plt.figure(200)
        plt.clf()
        plt.plot(teffG,loggG,'k+')
        plt.plot([teffb],[loggb],'rx')
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.ylabel(r'$log g$')
        plt.xlabel(r'$T_{\rm eff}$ [K]')
        plt.draw()
        
    return name, teffb , loggb,massb,radiusb
    
    '''
    files = pack['files']     
    glob = pack['glob'] # star global parameters
    
    
    glob[i,j],
    i: model index
    j: parameter index:
      0 : M_star
      1 : R_tot
      2 : L_tot
      3 : Z0
      4 : X0
      5 : alpha
      6 : X in CZ
      7 : Y in CZ
      8 : d2p
      9 : d2ro
      10 : age
      11 : wrot initial (global rotation velocity)
      12 : w_rot initial
      13 : g constante de la gravitaion
      14 : msun
      15 : rsun
      16 : lsol
      17 : Teff
      18 : log g
      19 : Tc temperature at the center
      20 : numax (scaling) [muHz]
      21 : deltanu (scaling) [muHz]
      22 : acoustic diameter [sec]
      23 : nuc, cutoff frequency, at the photosphere [muHz]
      24 : nuc, cutoff frequency, at the r=rmax [muHz]
      25 : deltaPI_1 [sec]
      26,27 : r1,r2 -> interval in radii on which the Brunt-Vaisala is integrated for the calculation of deltaPI
      28 : Xc
      29 : Yc
      30 : acoustic diameter [sec], computed on the basis of the .amdl file (can be oversampled)
      31 : acoustic depth of the Gamma1 bump associated with the first He ionization zone
      32 : acoustic depth of the Gamma1 bump associated with the second He ionization zone
      33 : acoustic depth of the base of the convective zone    

    references[i]: some references values:
            0: msun
            1: rsun
            2: lsun
            3: teff sun
            4: logg sun
            5: numax ref. (Mosser et al 2013)
            6: deltanu ref. (Mosser et al 2013)
            7: nuc sun (Jimenez 2006)
    '''

def prepare_spot_parameters(Star,Spot,Duration,seed=None,verbose=False):
    # gather together all the parameters used by the spot modelling library (spotintime)


    # mean rotation period in days
    prot =  Star['SurfaceRotationPeriod']
    if(prot<=0):
        raise sls.SLSError("surface rotation cannot be zero or negative")

    # light curve offset -> ??
    c0 = 0.0

    # inclination of the star in degrees, default value: 90
    incl = Star['Inclination']
    if(incl<=0):
        print('WARNING: inclination angle is zero, no spot signature possible!')

    # differential rotation (dimensionless), default value: 0
    domega = Spot['dOmega']

    #spots radii in degrees , default value: 2.5
    rayi = Spot['Radius']
    nspots = len(rayi)

    #spots latitudes in degrees
    lati =  Spot['Latitude']

    #spots longitudes in degrees, default value: 0
    longi = Spot['Longitude']

    # lifetime of the spot in days, default value: infinity
    taui = np.empty(nspots, dtype=object)
    for i in range(nspots):
        if( not isinstance(Spot['Lifetime'][i], str)):
            # if( Spot['Lifetime'][i].lowercase() != 'infinity'  ):
            taui[i] = Spot['Lifetime'][i]

    #time of maximum flux contrast of the spot in days, default value: 0
    ti0 = Spot['TimeMax']
    np.random.seed(seed)
    for i in range(nspots):
        if( ti0[i] <0):
            ti0[i] = np.random.random()*(Duration+4*taui[i]) -2*taui[i]
            if(verbose):
                print('TimeMax for spot #%i is drawn randomly and take the value: %f days' % (i,ti0[i]))
            # print(ti0[i],taui[i])

    #flux of the spot in units of unspotted stellar flux  (maximum constrast of the spot), default value: 0.7
    fsi =  Spot['Contrast']

    #limb darkening coefficient of the star, default: 0.59
    mue = Spot['MuStar']

    #limb darkening coefficient of the spot, default:   0.78
    mus = Spot['MuSpot']

    # Modulation period of the spot radii, default is None
    modulation = Spot['Modulation']

    params = np.empty(int(8 + 6 * nspots), dtype=object)
    params[0] = nspots
    params[1] = np.log(prot)
    params[2] = incl
    params[3] = domega
    params[4: 4 + nspots] = rayi
    params[4 + nspots:4 + 2 * nspots] = lati
    params[4 + 2 * nspots:4 + 3 * nspots] = longi
    params[4 + 3 * nspots:4 + 4 * nspots] = ti0
    params[4 + 4 * nspots:4 + 5 * nspots] = taui/prot # -> in unit of rotation period
    params[4 + 5 * nspots:4 + 6 * nspots] = fsi
    params[4 + 6 * nspots] = c0
    params[5 + 6 * nspots] = mue
    params[6 + 6 * nspots] = mus
    params[7 + 6 * nspots] = modulation
    return params

def generate_spot_LC(params,Sampling,Duration,TimeShift):
    # cadence = Instrument['Sampling']  # cadence in seconds

    # the spot Lightcurve (LC) is modelled at a cadence shorter than the rotation period but in general at a longer than the working cadence
    # then the short cadence LC (PSLS working cadence) is obtained by interpolating the long cadence LC
    prot = math.exp(params[1])

    # cadence used for the spot modelling (in days)
    cadence_lc = prot/100.  # in days

    # number of points in the long cadence  light curve
    n_lc  = int( math.ceil(Duration/cadence_lc) )

    # time in days (for the long cadence LC)
    t_lc = np.arange(n_lc)*cadence_lc

    nspots = int(params[0])

    # computes long cadence   light curve and returns in flx
    [flx_lc, inispots, ovl] = spotintime.paramtolc(params, t_lc, nspots)

    if ovl == 1:
        raise sls.SLSError("spot-Light curve not created because overlapping spots, please reconsider the spot parameters")

    # interpolate the spot model at the working cadence
    n =  int(Duration*86400./Sampling)
    t = (np.arange(n)*Sampling+ TimeShift)/86400.
    flx = np.interp(t,t_lc,flx_lc)

    # plt.figure(110)
    # plt.clf()
    # plt.plot(t,flx)
    # plt.draw()
    # plt.show()


    return flx


def usage():
    print ("usage: psls.py config.yaml")
    print ("      : ")
    print ("Options:")
    print ("-v : print program version")
    print ("-h : print this help")
    print ("-P : do some plots")
    print ("--pdf : the plots are saved as PDF otherwise as PNG (default)")
    print ("-V : verbose mode")
    print ("-f : save the LC associated with each individual camera, otherwise average over all the cameras (this is the default choice)")
    print ("-m : save the merged LC: LC from the same group of camera are averaged and then averaged LCs are merged(/interlaced) together")
    print ("-o <path> : output directory (the working directory is by default assumed)")
    print ("-M <number> : number of Monte-Carlo simulations performed")
    print ("--extended-plots : an extended set of plots are displayed (activates automatically  the -P option)")
    print("--psd : save the PSD associated with the averaged light-curve (averaged over all cameras) and the merged light-curve (with the option -m)")
    print('--hdf5 : save in a HDF5 file the mean light-curve (LC)  and the various simulation components and data')
    print('--proto-sas : the data are saved in a HDF5 file and in format compatible with the prototype SAS pipeline')


if(len(sys.argv)<2):
    usage()
    sys.exit(2)
try:
    opts,args = getopt.getopt(sys.argv[1:],"hvPVo:fmM:",["pdf","extended-plots","psd","hdf5","proto-sas"])

except getopt.GetoptError as err:
    print (str(err))
    usage()
    sys.exit(2)

Verbose = False
Plot = False
OutDir = '.'
FullOutput =  False  # single camera light-curves are saved 
MergedOutput = False  # LC from the same group of camera are averaged and  then averaged LC are merged(/interlaced)
Pdf = False
MC = False  # Monte-Carlo simulations on/off 
nMC = 1 # Number of  Monte-Carlo simulations
ExtendedPlots = False
SavePSD = False
SaveHDF5 = False
ProtoSAS = False # data saved in a format compatible with the prototype SAS pipeline

for o, a in opts:
    if o == "-h" :
        usage()
        sys.exit(1)
    elif o == "-v":
        print (__version__)
        sys.exit(1)
    elif o == "-V":
        Verbose = True
    elif o == "-P":
        Plot = True
    elif o == "-f":
        FullOutput = True
    elif o == "-m":
        MergedOutput = True
    elif o == "-o":
        OutDir = a
    elif o == "-M":
        MC = True
        nMC = int(a)
    elif o == "--pdf":
            Pdf = True    
    elif o == "--extended-plots":
        ExtendedPlots = True
        Plot = True
    elif o == "--psd":
        SavePSD = True
    elif o == "--hdf5":
        SaveHDF5 = True
    elif o == "--proto-sas":
        SaveHDF5 = True
        ProtoSAS = True
        MergedOutput = True
    else:
        print ("unhandled option %s" % (o))
        sys.exit(1)


nargs = len(args)
if nargs > 1 :
    print ("too many arguments")
    usage()
    sys.exit()

if nargs < 1 :
    print ("missing arguments")
    usage()
    sys.exit()

if(MC & Plot):
    print ("The options -M and -P are not compatible. ")    
    sys.exit()

if(MC & Verbose):
    print ("The options -M and -V are not compatible. ")    
    sys.exit()

if(Plot):   
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()

if(not SaveHDF5):
    if (FullOutput & MergedOutput):
        print ("The options -m and -f are not compatible. If you want both use the --hdf5 option")
        sys.exit()
  

config=args[0]

stream = open(config, 'r')    # 'document.yaml' contains a single YAML document.
if(parse_version(yaml.__version__)< parse_version("5.0")):
    cfg = yaml.load(stream)
else:
    cfg = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

OutDir = os.path.normpath(OutDir) + '/'

Star = cfg['Star'] 
StarModelType = Star['ModelType']
StarModelName = Star['ModelName']

Osc = cfg['Oscillations']
OscEnable = Osc['Enable'] 

StarID = Star['ID']
StarName = ("%10.10i") % StarID
StarTeff,StarLogg = Star['Teff'],Star['Logg']
StarES = Star['ES']
if(Verbose):
    print ('Star name: ' + StarName)

UP = StarModelType.lower() == 'up'

if (UP):
    DPI = Osc['DPI']
    q = Osc['q']
    numax = Osc['numax']
    delta_nu =  Osc['delta_nu']    
else:
    sls.numaxref= 3050.
    StarModelDir = Star['ModelDir']
    if(StarModelDir is None):
        StarModelDir = '.'
    StarModelDir = os.path.normpath(StarModelDir) + '/'
    StarMass, StarRadius = -1., -1.
    if ( (StarModelType.lower() == 'text') ):
        StarFreqFile = StarModelDir +  StarModelName
        StarFreqFileType = 0
        numax = Osc['numax']
        delta_nu =  Osc['delta_nu']
    elif ( (StarModelType.lower() == 'single') or (StarModelType.lower() == 'adipls') ):
        StarFreqFile =  StarModelDir + StarModelName
        _, extension = os.path.splitext(StarFreqFile)
        if(extension=='' ):
            StarFreqFile += '.gsm'
        StarFreqFileType = 1
        numax = -1.
        delta_nu = -1.
    elif(StarModelType.lower() == 'grid-old'): # old version of the model grid
        StarFreqFileType = 1
        numax = -1.
        delta_nu = -1.
        if(Verbose):
            print ('requested values:')
            print (('teff = %f ,log g = %f') % (StarTeff, StarLogg))
        StarModelName,StarTeff,StarLogg,StarMass,StarRadius = search_model(StarModelDir,StarES,StarLogg,StarTeff,verbose=Verbose,plot=Plot)
        if(Verbose):
            print ('closest values found:')
            print (('teff = %f ,log g = %f') % (StarTeff, StarLogg))   
        StarFreqFile = StarModelDir + StarModelName + '.gsm'

    elif(StarModelType.lower() == 'grid'): #  new version of the model grid
        StarFreqFileType = 2
        numax = -1.
        delta_nu = -1.
        StarModelFile = StarModelDir + StarModelName
        _, extension = os.path.splitext(StarModelFile)
        if(extension=='' or extension.lower() != '.hdf5'):
             StarModelFile += '.hdf5'
        if(Verbose):
            print ('requested values:')
            print (('teff = %f ,log g = %f') % (StarTeff, StarLogg))
        StarFreqFile,StarTeff,StarLogg,StarMass,StarRadius = search_model_hdf5(StarModelFile,StarES,StarLogg,StarTeff,verbose=Verbose,plot=Plot)
        if(Verbose):
            print ('closest values found:')
            print (('teff = %f ,log g = %f') % (StarTeff, StarLogg))
    else:
        print ("unhandled StarModelType: %s" % (StarModelType))
        sys.exit(1)

    logTeff = math.log10(StarTeff)
    SurfaceEffects = Osc['SurfaceEffects']
    if(SurfaceEffects and (StarFreqFileType>0)):
        
        if(  pip(StarTeff,StarLogg,[[5700.,4.6],[6700.,4.4],[6500.,3.9],[5700,3.9]]) == False):
            print ("surface effects: Teff and log g outside the table") 
            sys.exit(1) 

        # from Sonoi et al 2015, A&A, 583, 112  (Eq. 10 & 11)
        logma =  7.69 * logTeff -0.629 * StarLogg -28.5
        logb = -3.86*logTeff  + 0.235 * StarLogg + 14.2
    
        a =  - 10.**logma
        b = 10.**logb
        if(Verbose):
            print (('Surface effects parameters, a = %f ,b = %f') %  (a,b))
    
    else:
        a = 0.
        b = 1.


OutDir = os.path.normpath(OutDir) + '/'

Granulation = cfg['Granulation']
Granulation_Type = int(Granulation['Type'])
GranulationEnable = (Granulation['Enable']==1)
Transit = cfg['Transit']

Observation = cfg['Observation']
Gaps = Observation['Gaps']

Instrument = cfg['Instrument']
IntegrationTime = Instrument['IntegrationTime']

# initialization of the state of the RNG
MasterSeed = Observation['MasterSeed']

# Instrument parameters
GroupID = np.array( Instrument['GroupID'])
NGroup = len(GroupID)
GroupIDX  = GroupID-1

NCamera = Instrument['NCamera']  # Number of camera per group (1->6)

np.random.seed(MasterSeed)
seeds = np.random.randint(0, 1073741824 + 2,size=(NGroup*NCamera)*nMC+10*nMC)

#seeds
# 0: oscillation
# 1: systematics
# 2: spot
# 3: granulation
# 4: activity
# 5: gaps
if(OscEnable):
    if(Osc['Seed']>0):
        seeds[0] = int(Osc['Seed']) # seed not controlled by the master seed

if(GranulationEnable):
    if(Granulation['Seed']>0):
        seeds[3] = int(Granulation['Seed']) # seed not controlled by the master seed

GapsEnable = (Gaps['Enable']==1)
if(GapsEnable):
    InterQuarterGapDuration = float(Gaps['InterQuarterGapDuration']) # duration of the inter-quarter gaps (days)
    RandomGapDuration =   float(Gaps['RandomGapDuration'])    # duration of the random gaps  (minutes)
    RandomGapTimeFraction =  float(Gaps['RandomGapTimeFraction'])  # fraction (in %f) of the total time lost by the random gaps
    RandomGapStep = float(Gaps['RandomGapStep'])  # Random gap step in %
    PeriodicGapDuration =  float(Gaps['PeriodicGapDuration']) # duration of each  periodic gap  (minutes)
    PeriodicGapCadence =  float(Gaps['PeriodicGapCadence']) # Cadence of the periodic gap (in days)
    PeriodicGapJitter =  float(Gaps['PeriodicGapJitter'])  # Jitter in time the periodic gaps occur  (in days) ; uniform distribution assumed
    PeriodicGapStep = float(Gaps['PeriodicGapStep'])  # Periodic gap step in %
    if(Gaps['Seed']>0):
        seeds[4] = int(Gaps['Seed'])  # seed not controlled by the master seed

Sampling = Instrument['Sampling']

if( Sampling % 25 !=0 ):
    sls.SLSError('sampling must be a multiple of 25s')
QuarterDuration = np.array(Observation['QuarterDuration'])
Duration = np.sum(QuarterDuration) # total duration in days
NQuarter = len(QuarterDuration)
StarVMag = Star['Mag']

StarPMag = StarVMag - VmP(StarTeff)
StarVpMag = StarPMag + 0.34 # PLATO V reference magnitude (reference star of 6000K)

activity = None
Activity = cfg['Activity']
ActivityEnable = (Activity['Enable'] == 1)
if(ActivityEnable):
    activity = (Activity['Sigma'],Activity['Tau'])
    if(Activity['Seed']>0):
        seeds[4] = int(Activity['Seed']) # seed not controlled by the master seed

spot = None
Spot = Activity['Spot']
SpotEnable = (Spot['Enable'] == 1)
if(SpotEnable):
    if(Spot['Seed']>0):
        seeds[2] = int(Spot['Seed'])  # seed not controlled by the master seed
    spot = prepare_spot_parameters(Star,Spot,Duration,seed=seeds[2],verbose=Verbose)

External = cfg['External']
ExternalEnable = (External['Enable'] == 1)


Flare = Activity['Flare']
FlareEnable = (Flare['Enable'] == 1)

if(FlareEnable):
    FlareUpDown = Flare['UpDown']  # Ratio of the time taken for the flow to rise to the time taken for the flow to fall
    FlareAmplitude = Flare['Amplitude']  # mean flares amplitudes (ppm)
    FlareMeanPeriod = Flare['MeanPeriod']  # mean period btw  2 flares (days)
    FlareMeanDuration = Flare['MeanDuration']   # mean flare duration (days). If negative,  FlareMenDuration = FlareMeanPeriod/5
    FlareDurationDispersion = Flare['DurationDispersion'] # dispersion in the flare duration (days). If negative,  FlareDurationDispersion = FlareDurationDispersion/20

    if (Flare['Seed'] > 0):
        seeds[5] = int(FlareEnable['Seed'])  # seed not controlled by the master seed

if(Verbose):
    print ('V magnitude: %f' % StarVMag)
    print ('P magnitude: %f' % StarPMag)
    print ('Reference V PLATO magnitude (6000 K): %f' % StarVpMag)

if (UP):
    # Simulated stellar signal, noise free (nf), UP
    time,ts,f,ps,_,mps_nf_osc,mps_nf_granulation,mps_nf_activity,opar,_ =  sls.gen_up(StarID, numax,Sampling,Duration,
                                             StarVMag,delta_nu =  delta_nu, mass = -1. , seed =seeds[0] , pn_ref = 0.,
                                             wn_ref= 0., mag_ref = 6.,verbose = Verbose, teff = StarTeff, DPI = DPI,
                                             q = q, GST = Granulation_Type, incl = Star['Inclination'],
                                             rot_core_f = Star['CoreRotationFreq'], rot_period_sur =  Star['SurfaceRotationPeriod'],
                                             path = OutDir,
                                              oscillation = True,granulation=GranulationEnable, activity=activity)

else:
    # Simulated stellar signal, noise free (nf)
    time,ts,f,_,_,mps_nf_osc,mps_nf_granulation,mps_nf_activity,opar,_ = sls.gen_osc_spectrum(StarID,StarFreqFile,
                                StarTeff,StarMass, StarRadius,Sampling,Duration,StarVMag,verbose=Verbose,seed=seeds[0],
                                mag_ref=StarVMag,pn_ref= 0.,wn_ref= 0., a=a,b=b,plot=0, rot_period_sur =  Star['SurfaceRotationPeriod'] ,
                                                                                              rot_core_f = Star['CoreRotationFreq'],
                                            incl = Star['Inclination'] , path = OutDir , oscillation = True,
                                                      type = StarFreqFileType,numax=numax,deltanu=delta_nu,
                                                       GST=Granulation_Type,granulation=GranulationEnable,
                                                       activity=activity)

TransitEnable = (Transit['Enable']==1)
if(TransitEnable):
    SampleNumber = time.size
    StarRadius = opar[0]['radius']*sls.rsun*1e-5 # in km
    PlanetRadius = Transit['PlanetRadius'] * jupiterRadius  # in km
    p = PlanetRadius / StarRadius
    _, z,transit_dates,transit_phases = generateZ(Transit['OrbitalPeriod']*86400., Transit['PlanetSemiMajorAxis']*ua2Km,
              StarRadius,Sampling, IntegrationTime, 0. , SampleNumber,
              Transit['OrbitalAngle']*math.pi/180., p)
    ## gamma = [.25, .75]
    gamma = np.array(Transit['LimbDarkeningCoefficients'],dtype=np.float64)
    if  len(gamma) == 4:  transit = tr.occultnonlin(z, p, gamma)
    else:  transit = tr.occultquad(z, p, gamma, verbose=Verbose)
    if(Verbose):
        print (('Star radius [solar unit]: %f') % ( opar[0]['radius'])) 
        print ("Planet Radius/ Star Radius = {0:}".format(p))
        print (("Transit depth: %e" ) % (np.max(transit)/np.min(transit)-1.))
    if(Plot):
        plt.figure(110)
        plt.clf()
        plt.title(StarName+ ', transit')
        plt.axvline(x=transit_phases[0],color= 'r',ls='--')
        plt.axvline(x=transit_phases[1],color='r',ls='--')
        plt.axvline(x=transit_phases[2],color='r',ls='--')
        plt.plot(time/86400.,(transit-1.)*100.)
        plt.ylabel('Flux variation [%]')
        plt.xlabel('Time [days]')
        plt.draw()
        # plt.figure(111)
        # plt.clf()
        # plt.title(StarName+ ', transit')
        # plt.plot(time/86400.,z,'k.')
        # plt.axvline(transit_phases[0],'r.')
        # plt.axvline(transit_phases[1],'r+')
        # plt.axvline(transit_phases[2],'r.')
        # plt.xlabel('Time [days]')
        # plt.draw()
        # plt.show(block=True)

Systematics = Instrument['Systematics']
SystematicDataVersion = int(Systematics['Version'])
DataSystematic = None
SystematicsEnable = (Systematics['Enable']==1)
if(SystematicsEnable):
    if(Systematics['Seed']>0): # seed NOT controlled by the master seed
        seeds[1] = int(Systematics['Seed'])
    if(SystematicDataVersion>0):
        DriftLevel = (Systematics['DriftLevel']).lower() 
    else:
        DriftLevel = None
    if(SystematicDataVersion<2): # check if all quarters last 90 days:
        m = (np.abs(QuarterDuration-90.) > 0.1)
        if(m.sum()>0):
            raise sls.SLSError("When SystematicDataVersion<2, all quarters shall last excatly 90 days")

    DataSystematic = sls.ExtractSystematicDataMagRange(
        Systematics['Table'],StarVpMag,version=SystematicDataVersion,DriftLevel=DriftLevel,
        Verbose=Verbose,seed=seeds[1])
    
# Total white-noise level ppm/Hz^(1/2), for a each single Camera
RandomNoise =  Instrument['RandomNoise']
RandomNoiseEnable = (RandomNoise['Enable']==1)
if RandomNoiseEnable:
    if (RandomNoise['Type'].lower() == 'user'):
        NSR = float(RandomNoise['NSR'])
    elif (RandomNoise['Type'].lower() == 'plato_scaling'):
        if( (StarPMag<NSR_Pmag[0]-0.25) or (StarPMag>NSR_Pmag[-1]+0.25) ):
            print ("Warning: Star magnitude out of range, boundary NSR value is assumed")
        NSR = np.interp(StarPMag, NSR_Pmag, NSR_values, left=NSR_values[0], right=NSR_values[-1])
    elif (RandomNoise['Type'].lower() == 'plato_simu'):
        if(DataSystematic is None):
            raise sls.SLSError("RandomNoise: when Type=PLATO_SIMU, systematic errors must also be activated")   
        if(SystematicDataVersion<1):
            raise sls.SLSError("RandomNoise: when Type=PLATO_SIMU, data version for systematic errors must be >0")   
        NSR = -1.
    else:
        raise sls.SLSError("unknown RandomNoise type: "+ RandomNoise['Type'] + ' Can be either USER, PLATO_SCALING or PLATO_SYSTEMATICS')        
else:
    NSR = 0.
W =  NSR*math.sqrt(3600.) # ppm/Hr^(1/2) -> ppm/Hz^(1/2)
opar[1]['white_noise'] = W #  ppm/Hz^(1/2)
dt = opar[1]['sampling'] 
nyq = opar[1]['nyquist']

if(Verbose and W>=0.):
        print ('NSR for one camera: %f [ppm.sqrt(hour)]' % NSR)
        print ('Total white-noise for one camera [ppm/Hz^(1/2)]: %f' % W)
        print ('Total white-noise at sampling time for one camera [ppm]: %f' % (W/math.sqrt(Sampling)))
if(Verbose):
        print ('Nyquist frequency [muHz]: %f' % nyq)
        print ('frequency resolution [muHz]: %f' % f[0])
TimeShift = Instrument['TimeShift']

if(ExternalEnable):
    # loading the external component
    filepath = External['FilePath']
    if(Verbose): print(f'loading {filepath}')
    ExternalData = np.loadtxt(filepath)
    ExternalDuration = (ExternalData[-1,0] - ExternalData[0,0])/86400. # duration in days
    if(Verbose): print(f'duration of the external component: {ExternalDuration:.2f} days')
    # checking the time coverage
    if(ExternalDuration<Duration):
        raise sls.SLSError("The external component is too short, it shall cover the simulation duration")

nt = time.size
full_ts = np.zeros((nt,NGroup,NCamera,5))
nu = np.fft.fftfreq(nt,d=dt)[0:nt//2] * 1e6 # frequencies, muHz
nnu = nu.size
spec = np.zeros((NGroup,nu.size))
dnu = nu[1]
time += IntegrationTime/2.
if(FlareEnable):
    FlareLC = AddFlare(time,FlareMeanPeriod,FlareUpDown,FlareAmplitude,FlareMeanDuration,FlareDurationDispersion,seeds[5])

if(SystematicsEnable): full_ts_SC = np.zeros((nt,NGroup,NCamera,2)) # systematic error only
if(SaveHDF5):
    if(SpotEnable): spot_ts = np.zeros((nt,NGroup,2))
    if (TransitEnable): transit_ts = np.zeros((nt,NGroup,2))
    stellar_ts = np.zeros((nt,NGroup,2))
    if (RandomNoiseEnable): random_ts = np.zeros((nt,NGroup,NCamera,2))
    if(GranulationEnable): granulation_ts = np.zeros((nt,NGroup,2))
    if(ActivityEnable): activity_ts = np.zeros((nt,NGroup,2))
    if(OscEnable): osc_ts = np.zeros((nt,NGroup,2))
    if(FlareEnable): flares_ts = np.zeros((nt,NGroup))



for iMC in range(nMC):
    for i in range(NGroup):
        if(Verbose):
            print (('Group: %i') % (i))
        # simulating stellar signal
        # if(nMC>0): seed_i = seeds[10*iMC] # ?? should the stellar signal change for the different MC realzation ???
        # else:
        seed_i = seeds[0]
        group_idx = GroupIDX[i]
        time_i, ts_osc , ps_i, _ =  sls.mpsd2rts(f*1e-6,mps_nf_osc*1e6,seed=seed_i,time_shift=group_idx*TimeShift)
        time_i += IntegrationTime/2.
        if(OscEnable):
            delta_ts_i = ts_osc*1e-6
        else:
            delta_ts_i = 0.
        if(GranulationEnable):
            seed_i = seeds[3]
            _, ts_granulation, _, _ = sls.mpsd2rts(f * 1e-6, mps_nf_granulation * 1e6, seed=seed_i, time_shift=group_idx * TimeShift)
            delta_ts_i += ts_granulation*1e-6
        if(ActivityEnable):
            seed_i = seeds[4]
            _, ts_activity, _, _ = sls.mpsd2rts(f * 1e-6, mps_nf_activity * 1e6, seed=seed_i,time_shift=group_idx * TimeShift)
            delta_ts_i += ts_activity*1e-6
        if(SpotEnable):
            spot_component = generate_spot_LC(spot,Sampling,Duration,group_idx*TimeShift)
            if (SaveHDF5):
                spot_ts[:,i,1] = spot_component
        if (TransitEnable):
            p = PlanetRadius / StarRadius
            _, z,_,_ = generateZ(Transit['OrbitalPeriod'] * 86400., Transit['PlanetSemiMajorAxis'] * ua2Km,
                             StarRadius, Sampling, IntegrationTime, group_idx * TimeShift, SampleNumber,
                             Transit['OrbitalAngle'] * math.pi / 180., p)
            if (len(gamma) == 4):
                transit_component = tr.occultnonlin(z, p, gamma)
            else:
                transit_component = tr.occultquad(z, p, gamma, verbose=Verbose)
            if (SaveHDF5):
                transit_ts[:,i,1] = transit_component
        if(ExternalEnable):
            external_component = np.interp(time_i, ExternalData[:, 0] - ExternalData[0, 0], ExternalData[:, 1])
            external_component = 1. + external_component*1e-6
        if(FlareEnable):
            ts_flare =  np.interp(time_i,time,FlareLC)
            if(SaveHDF5):
                flares_ts[:,i] = ts_flare
            # plt.plot(ts_flare)
            # plt.show(block=True)
        if (SaveHDF5):
            stellar_ts[:,i,0] = time_i
            if(OscEnable):
                osc_ts[:,i,0] = time_i
                osc_ts[:, i, 1] = ts_osc
            if(GranulationEnable):
                granulation_ts[:, i, 0] = time_i
                granulation_ts[:,i,1] = ts_granulation
            if(ActivityEnable):
                activity_ts[:, i, 0] = time_i
                activity_ts[:,i,1] = ts_activity
            stellar_ts[:,i,1] = (1. + delta_ts_i )
            if(TransitEnable):
                transit_ts[:, i, 0] = time_i
                stellar_ts[:,i,1] *= transit_component
            if(SpotEnable):
                spot_ts[:, i, 0] = time_i
                stellar_ts[:,i,1] *= spot_component
            if(ExternalEnable):
                stellar_ts[:,i,1] *= external_component
            if(FlareEnable):
                stellar_ts[:,i,1] *= ts_flare

        for j in range(NCamera):
            if(SystematicsEnable): full_ts_SC[:,i,j,0] = time_i
            full_ts[:,i,j,0] = time_i
            if(SaveHDF5 and RandomNoiseEnable):
                random_ts[:,i,j,0] = time_i
            ts = (1. + delta_ts_i )
            seed_idx = i*NCamera+j+iMC*(NCamera*NGroup)+10*nMC
            np.random.seed(seeds[seed_idx])
            delta_ts_i_j = 0.
            if(SystematicsEnable):
                # adding systematic errors
                if(Verbose):
                    print('Generating systematic errors for camera %i of group %i' % (j,GroupID[i]))
                    # print(iMC,j,i,seeds[seed_idx])
                tsSC,rawLCvar,_,_, flag= sls.SimSystematicError(Sampling,nt,DataSystematic,group_idx*NCamera+j,QuarterDuration,
                                                                 seed=seeds[seed_idx],
                                                                 version=SystematicDataVersion,
                                                                 Verbose=False)
                if(Verbose):
                    p2p = (np.max(tsSC)/np.min(tsSC)-1.)*100
                    print('      peak to peak variation [%%]: %f ' %p2p)
                    print('      number of mask updates: %i' % (np.sum(flag)-1))
                full_ts[:,i,j,4] = flag
                delta_ts_i_j += (tsSC-1.)
            else:
                rawLCvar = 1.
                tsSC = 1.

            random_component = 0.
            if(RandomNoiseEnable):
                if(Verbose):
                    print('Generating random noise for camera %i of group %i' % (j,GroupID[i]))
                if(W>0.):
                    # adding random noise
                    random_component = np.random.normal(0.,W*1e-6/math.sqrt(Sampling),size=nt)
                elif(W<0.):
                    # adding random noise from LC variance
                    random_component = np.random.normal(0.,1.,size=nt)*np.sqrt(rawLCvar)
                if(Verbose): print(f'      noise level: {np.std(random_component)*1e6/np.sqrt(3600./Sampling):.2f} ppm.hr^(1/2)')
                delta_ts_i_j += random_component

            ts *= (1. + delta_ts_i_j)
            if(SaveHDF5 and RandomNoiseEnable):
                random_ts[:,i,j,1] = random_component*1e6
            if(TransitEnable):
                ts *= transit_component
            if(SpotEnable):
                ts *= spot_component
            if(ExternalEnable):
                ts *= external_component
            if (FlareEnable):
                ts *= ts_flare
            full_ts[:,i,j,1] = ts
            full_ts[:,i,j,2] = GroupID[i] # group number
            full_ts[:,i,j,3] = j+1 # camera number (within a given group)
            if(SystematicsEnable): full_ts_SC[:,i,j,1] = tsSC # systematics only


    # add gaps
    if(GapsEnable):
        if(Verbose):
            print('Gaps are added')
        if(Verbose):
            print(f'Number of quarters: {NQuarter}')

        time_days = np.arange(nt)*Sampling/86400.
        gaps_step = np.zeros(nt)
        is_gap1 = np.zeros(nt, dtype='bool')
        if(InterQuarterGapDuration>0.):
            start_quarter = 0.
            for iq in range(NQuarter):
                is_gap1 = is_gap1 | ((start_quarter <= time_days) & (time_days <= start_quarter + InterQuarterGapDuration))
                start_quarter += QuarterDuration[iq]

        total_duration = time_days[-1] - time_days[0]
        # Random interruptions
        is_gap2 = np.zeros(nt, dtype='bool')
        if(RandomGapDuration>0. and RandomGapTimeFraction>0.):
            InterruptionDurationDays = RandomGapDuration/(24.*60.)
            ni = int(math.floor(total_duration/InterruptionDurationDays*RandomGapTimeFraction/100.))
            if(Verbose):
                print(f"Total number of random interruptions: {ni}")
                print(f"Number of random interruptions per quarter: {ni/NQuarter}")

            np.random.seed(seeds[4])
            t0_gaps = np.random.uniform(time_days[0] , time_days[0] + total_duration, ni)
            t0_gaps_start = t0_gaps - InterruptionDurationDays/2.
            t0_gaps_end = t0_gaps + InterruptionDurationDays/2.
            k = 0
            for t1,t2 in zip(t0_gaps_start,t0_gaps_end):
                is_gap2 = is_gap2 | (  (time_days>=t1)  & ( time_days<t2)  )
                sign = (-1) ** (k)
                gaps_step[time_days>=t2] += (RandomGapStep*1e-2)*sign
                k += 1

        # Periodic interruptions
        is_gap3 = np.zeros(nt, dtype='bool')
        if(PeriodicGapDuration>0. and PeriodicGapCadence>0.):
            InterruptionDurationDays = PeriodicGapDuration/(24.*60.)
            ni = int(math.floor(total_duration/PeriodicGapCadence))
            if(Verbose):
                print(f"Total number of periodic interruptions: {ni}")
                print(f"Number of periodic interruptions per quarter: {ni/NQuarter}")

            def normal_restricted(size,vmin,vmax):
                """
                Normal distribution of zero mean and a 1-sigma of ine with
                limited values (vmin,vmax)

                :param size:
                :param vmin:
                :param vmax:
                :return:
                """
                x = np.random.normal(0.,1.,size=size)
                x = x.clip(min=vmin,max=vmax)
                return x

            gaps_jitter = normal_restricted(ni,-3.,3.)*(PeriodicGapJitter/24.)
            t0_gaps = (np.arange(ni)+1)*PeriodicGapCadence + gaps_jitter
            t0_gaps_start = t0_gaps - InterruptionDurationDays/2.
            t0_gaps_end = t0_gaps + InterruptionDurationDays/2.
            k = 0
            for t1,t2 in zip(t0_gaps_start,t0_gaps_end):
                is_gap3 = is_gap3 | (  (time_days>=t1)  & ( time_days<t2)  )
                sign = (-1) ** (k)
                gaps_step[time_days>=t2] += (PeriodicGapStep*1e-2)*sign
                k+= 1

        if (ExtendedPlots):
            plt.figure(300)
            plt.title('Gaps')
            if(RandomGapDuration>0. and RandomGapTimeFraction>0.): plt.plot(time_days,is_gap2,'r.',label='random')
            if (PeriodicGapDuration > 0. and PeriodicGapCadence > 0.):
                plt.plot(time_days,is_gap3,'b.',label='periodic')
            if(InterQuarterGapDuration>0.): plt.plot(time_days,is_gap1,'k.',label='Inter-quarter')
            plt.xlabel('Time [days]')
            plt.legend()
            plt.figure(301)
            plt.title('Gap steps [%]')
            plt.plot(time_days,gaps_step*1e2 , 'k.')
            plt.xlabel('Time [days]')



        is_gap = is_gap1 | is_gap2 | is_gap3
        not_gap = (is_gap==False)
        if (Verbose):
            duty_cycle = not_gap.sum()/nt*100.
            print(f'Duty cycle: {duty_cycle} [%]')

        full_ts = full_ts[not_gap]
        for i in range(NGroup):
            for j in range(NCamera):
                full_ts[:,i,j,1]  *=  (1. + gaps_step[not_gap] )

        if (SystematicsEnable): full_ts_SC = full_ts_SC[not_gap]
        nt = np.sum(not_gap)
        if (SaveHDF5):
            if(SpotEnable): spot_ts =  spot_ts[not_gap]
            if(TransitEnable): transit_ts =  transit_ts[not_gap]
            stellar_ts =  stellar_ts[not_gap]
            if(RandomNoiseEnable): random_ts =  random_ts[not_gap]
            if(ActivityEnable): activity_ts = activity_ts[not_gap]
            if(OscEnable): osc_ts = osc_ts[not_gap]
            if(GranulationEnable): granulation_ts = granulation_ts[not_gap]
            if(FlareEnable): flares_ts = flares_ts[not_gap]

    # relative flux variation, in ppm
    for i in range(NGroup):
        for j in range(NCamera):
            full_ts[:, i, j, 1] = (full_ts[:,i,j,1]-1.)*1e6
            if (SystematicsEnable):
                full_ts_SC[:,i,j,1] = (full_ts_SC[:,i,j,1]-1.)*1e6

    # LC averaged over the camera groups
    single_ts = np.zeros((nt,3))
    single_ts[:,0] = np.sum(full_ts[:,:,:,0],axis=(1,2))/float(NGroup*NCamera)
    single_ts[:,1] = np.sum(full_ts[:,:,:,1],axis=(1,2))/float(NGroup*NCamera)
    single_ts[:,2] = np.sum(full_ts[:,:,:,4],axis=(1,2))
    if (SystematicsEnable):
        single_ts_SC = np.sum(full_ts_SC[:,:,:,1],axis=(1,2))/float(NGroup*NCamera)

    if(SavePSD or Plot):
        single_nu,single_psd = LS(single_ts[:,0],single_ts[:,1],Sampling) # Lomb-Scargle periodogram
        single_nu *= 1e6 # Hz->muHz
        single_psd *= 1e-6 # ppm^2/Hz -> ppm^2/muHz

    if(Verbose):
            high_freq_var = np.var(  np.diff(single_ts[:,1]) )/2.
            high_freq_sig = math.sqrt(high_freq_var)
            print (f'standard deviation of the averaged light-curves: {high_freq_sig} ')
            high_freq_nsr = high_freq_sig/math.sqrt(3600./Sampling)
            high_freq_nsr = high_freq_sig/math.sqrt(3600./Sampling)
            print(f'corresponding NSR = {high_freq_nsr}  ppm.hr^(1/2)')

    if(MC):
        StarName = ("%7.7i%3.3i") % (StarID,iMC)
    else:
        StarName = ("%10.10i") % StarID
            
    if(SaveHDF5):
        fname = OutDir + StarName+ '.hdf5'
    else:
        fname = OutDir + StarName + '.dat'
    if(Verbose):
        print ('saving the simulated light-curve as: %s' % fname)
    
    def ppar(par):
        n = len(par)
        i = 0
        s = ''
        for u in list(par.items()):
            s += ' %s = %g' % u
            if (i < n-1):
                s += ', '
            i += 1
        return s
    
    hd = ''
    hd += ('StarID = %10.10i\n') % (StarID)  
    hd += ("Master_seed = %i\n") % (MasterSeed)
    hd += ("Version = %7.2f\n") % (__version__)

    if(MergedOutput):
        merged_ts = np.zeros((nt, NGroup, 4))
        for G in range(NGroup):
            merged_ts[:, G, 0] = rebin1d(full_ts[:, G, :, 0].flatten(), nt) / float(NCamera)
            merged_ts[:, G, 1] = rebin1d(full_ts[:, G, :, 1].flatten(), nt) / float(NCamera)
            merged_ts[:, G, 2] = GroupID[G]
            merged_ts[:, G, 3] = rebin1d(full_ts[:, G, :, 4].flatten(), nt)
        merged_ts = merged_ts.reshape((nt * NGroup, 4))

    if(SavePSD and MergedOutput):
        merged_nu,merged_psd = LS(merged_ts[:,0], merged_ts[:,1],Sampling) # Lomb-Scargle periodogram
        merged_nu *= 1e6 # Hz-> muHz
        merged_psd  *= 1e-6 # ppm^2/Hz -> ppm^2/muHz

    full_hd = hd + '# star parameters:\n'
    full_hd += (' teff = %f ,logg = %f\n') % (StarTeff, StarLogg)
    full_hd += ppar(opar[0])
    full_hd += '\n# observations parameters:\n'
    full_hd += ppar(opar[1])
    full_hd += '\n# oscillation parameters:\n'
    full_hd += ppar(opar[2])
    if(GranulationEnable):
        full_hd += '\n# granulation parameters:\n'
        full_hd += ppar(opar[3])
    if (ActivityEnable):
        full_hd += '\n# activity parameters:\n'
        full_hd += ppar(opar[4])

    fd = open( OutDir + StarName + '.txt', 'w')
    fd.write(full_hd)
    fd.close()

    if(SaveHDF5):
        hdf5file = h5py.File(fname, 'w')
        # save the meta-data
        hdf5file.attrs['StarID'] = StarID
        hdf5file.attrs['MasterSeed'] = MasterSeed
        hdf5file.attrs['Version'] = __version__
        hdf5file.attrs['StarID'] = StarID
        hdf5file.attrs['GranulationEnable'] = GranulationEnable
        hdf5file.attrs['ActivityEnable'] = ActivityEnable
        hdf5file.attrs['OscEnable'] = OscEnable
        hdf5file.attrs['SystematicsEnable'] = SystematicsEnable
        hdf5file.attrs['TransitEnable'] = TransitEnable
        hdf5file.attrs['SpotEnable'] = SpotEnable
        hdf5file.attrs['RandomNoiseEnable'] = RandomNoiseEnable
        hdf5file.attrs['FlareEnable'] = FlareEnable
        hdf5file.attrs['OscSeed'] = seeds[0]
        hdf5file.attrs['SystematicsSeed'] = seeds[1]
        hdf5file.attrs['SpotSeed'] = seeds[2]
        hdf5file.attrs['GranulationSeed'] = seeds[3]
        hdf5file.attrs['ActivitySeed'] = seeds[4]
        hdf5file.attrs['FlareSeed'] = seeds[5]
        for o in opar:
            for key, val in o.items():
                hdf5file.attrs.create(key,val)

        # save the setup
        hdf5file.attrs['Setup'] = yaml.dump(cfg)

        # save the data
        dset1 = hdf5file.create_dataset('MEAN_LC', data=single_ts[:,1:])
        if(FullOutput):
            dset2 = hdf5file.create_dataset('INDIVIDUAL_LC', data=full_ts[:,:,:,1:])
        if(MergedOutput and not ProtoSAS):
            dset3 = hdf5file.create_dataset('MERGED_LC', data=merged_ts[:,1:])
        if(RandomNoiseEnable): dset4 = hdf5file.create_dataset('RANDOM_LC', data=random_ts[:,:,:,1:].squeeze(axis=3))
        dset5 = hdf5file.create_dataset('STELLAR_LC', data=stellar_ts[:,:,1:].squeeze(axis=2))
        if(SystematicsEnable): dset6 = hdf5file.create_dataset('SYSTEMATICS_LC', data=full_ts_SC[:,:,:,1:].squeeze(axis=3))
        if(SpotEnable): dset7 = hdf5file.create_dataset('SPOT_LC', data=spot_ts[:,:,1:].squeeze(axis=2))
        if(FlareEnable): dset11 = hdf5file.create_dataset('FLARES_LC', data=flares_ts[:,:])
        if(TransitEnable and not ProtoSAS): dset8 = hdf5file.create_dataset('TRANSIT_LC', data=transit_ts[:,:,1:].squeeze(axis=2))
        if(SavePSD):
            PSD = np.array([single_nu,single_psd]).transpose()
            dset9 = hdf5file.create_dataset('MEAN_PSD', data=PSD)
            if (MergedOutput):
                PSD = np.array([merged_nu,merged_psd]).transpose()
                dset10 = hdf5file.create_dataset('MERGED_PSD', data=PSD)
        if(ActivityEnable):
            dset11 = hdf5file.create_dataset('ACTIVITY_LC', data=activity_ts[:,:,1:].squeeze(axis=2))
        if(OscEnable):
            dset12 = hdf5file.create_dataset('OSCILLATIONS_LC', data=osc_ts[:,:,1:].squeeze(axis=2))
        if(GranulationEnable):
            dset13 = hdf5file.create_dataset('GRANULATION_LC', data=granulation_ts[:,:,1:].squeeze(axis=2))
        dset14 = hdf5file.create_dataset('TIME', data=stellar_ts[:,:,0])
        if(MergedOutput and not ProtoSAS):
            dset15 = hdf5file.create_dataset('TIME_MERGED', data=merged_ts[:,0])
        dset16 = hdf5file.create_dataset('TIME_MEAN', data=single_ts[:,0])
        if(ProtoSAS):
            PRODUCTS = hdf5file.create_group("PRODUCTS")
            EXTERNAL = hdf5file.create_group("EXTERNALS")
            DP1_LC_MERGED = EXTERNAL.create_group("DP1_LC_MERGED")
            DP1_LC_MERGED_FLAGS =  EXTERNAL.create_group("DP1_LC_MERGED_FLAGS")
            DP1_LC_MERGED.create_dataset('flux', data=(merged_ts[:,1]*1e-6+1.) )
            t = (merged_ts[:,0]/86400.) # time in days
            DP1_LC_MERGED.create_dataset('time', data=t)
            quarter_num = np.zeros(t.size,dtype=np.int32)
            start_quarter = 0.
            for iq in range(NQuarter):
                m = (start_quarter<=t) & (t<start_quarter+QuarterDuration[iq]+25.)
                quarter_num[m] = iq + 1
                start_quarter += QuarterDuration[iq]
            DP1_LC_MERGED.create_dataset('quarter', data=quarter_num)
            flag =  np.array(merged_ts[:,2],dtype=np.int32)
            DP1_LC_MERGED_FLAGS.create_dataset('flag', data=flag)
            IDP_EAS_QC_TARGET_TCE_DETECTION = EXTERNAL.create_group("IDP_EAS_QC_TARGET_TCE_DETECTION")
            METADATA = hdf5file.create_group("METADATA")
            METADATA.attrs.create('star_id', StarID, dtype=int)
            METADATA.attrs.create('cadence',Sampling,dtype=float)
            METADATA.attrs.create('magnitude', StarVMag, dtype=float)
            METADATA.attrs.create('LC_len',Duration, dtype=float)
            description = 'This light curve was simulated by PSLS'
            METADATA.attrs.create('description', description)
            if(TransitEnable):
                IDP_EAS_TRANSIT_REMOVAL_KIT = EXTERNAL.create_group("IDP_EAS_TRANSIT_REMOVAL_KIT")
                Eclipse = IDP_EAS_TRANSIT_REMOVAL_KIT.create_group('Eclipse')
                Flux_curve = IDP_EAS_TRANSIT_REMOVAL_KIT.create_group('Relative_flux_curve')
                Flux_curve.create_dataset('flux', data=transit_ts[:,:,1].reshape((nt*NGroup)))
                TCE = IDP_EAS_TRANSIT_REMOVAL_KIT.create_group('TCE')
                TCE.create_dataset('Period',data=np.array([Transit['OrbitalPeriod']]))
                Eclipse.create_dataset('t0', data=np.array([transit_phases[0]]))
                Eclipse.create_dataset('t1', data=np.array([transit_phases[1]]))
                Eclipse.create_dataset('t2', data=np.array([transit_phases[2]]))
                IDP_EAS_QC_TARGET_TCE_DETECTION.create_dataset('flag', data=True)
            else:
                IDP_EAS_QC_TARGET_TCE_DETECTION.create_dataset('flag', data=False)
        hdf5file.close()
    else:
        if(SavePSD):
            np.savez(OutDir + StarName + '-averagedLC-PSD', nu=single_nu, psd=single_psd)
            if (MergedOutput):
                np.savez(OutDir + StarName+ '-mergedLC-PSD',nu=merged_nu,psd=merged_psd)
        if(FullOutput):
            full_ts = full_ts.reshape((nt*NGroup*NCamera,5))
            np.savetxt(fname,full_ts,fmt='%12.2f %20.15e %1i %1i %1i',header=hd + '\nTime [s], Flux variation [ppm], Group ID, Camera ID, Flag')
        elif(MergedOutput):
            np.savetxt(fname,merged_ts,fmt='%12.2f %20.15e %1i %1i',header=hd + '\nTime [s], Flux variation [ppm], Group ID, Flag')
        else:
            np.savetxt(fname,single_ts,fmt='%12.2f %20.15e %1i',header=hd + '\nTime [s], Flux variation [ppm], Flag')


if(Plot):
    # releasing unused variables
    merged_ts = None
    if(not ExtendedPlots):
        full_ts_SC = None
        full_ts = None

    plt.figure(100)
    plt.clf()
    plt.title(StarName+ ' PSD')
    plt.plot(single_nu,single_psd,'grey',label='simulated (raw)')
    win = opar[2]['numax']/100.
    m = int(round(win/single_nu[0]))
    p = int(single_nu.size/m)
    num = rebin1d(single_nu[0:p*m],p)/float(m)
    if (SystematicsEnable):
        _,psdsc = LS(single_ts[:,0],single_ts_SC,Sampling)
        psdsc *= 1e-6
        psdscm = rebin1d(psdsc[0:p*m],p)/float(m)
        plt.plot(single_nu,psdsc,'b:',label='systematics')

    psdm = rebin1d(single_psd[0:p*m],p)/float(m)
    mps_nf = mps_nf_osc
    if(not OscEnable):
        mps_nf[:] = 0.
    if(GranulationEnable):
        mps_nf += mps_nf_granulation
    if(ActivityEnable):
        mps_nf += mps_nf_activity
    if(W>0.):
        psdr = np.ones(mps_nf.size)*W**2*1e-6/NCamera/NGroup # random noise component
    else:
        psdr = 0.
    psdme = 0.5*mps_nf +  psdr # mean expected PSD for all camera
    
    plt.plot(num,psdm,'k',lw=2,label='simulated (mean)') # simulated spectrum, all camera
    if (SystematicsEnable):
            plt.plot(num,psdscm,'m',label='systematics (mean)')

    # factor 1/2 to convert  the PSD from single-sided to double-sided PSD
    ## plt.plot(f[1:], psdme[1:] ,'b',lw=2,label='star+instrument')  # All Camera
    plt.plot(f[1:], 0.5*mps_nf[1:],'r',lw=2,label='star') # noise free
    if(W>0.):
        plt.plot(f[1:], psdr[1:],'g',lw=2,label='random noise') # all Camera
#    plt.plot(f[1:], 0.5*( (mps_SC[1:]  - 2*W**2*1e-6/NCamera) /(NGroup)),'m',lw=2,label='systematics') # all Camera
    
    fPT,psdPT = platotemplate(Duration, dt=1., V=11., n=NCamera*NGroup, residual_only=True)
    plt.plot(fPT[1:]*1e6, psdPT[1:]*1e-6,'k',ls='--',lw=2,label='systematics (requierements)')
    
    plt.loglog()
    plt.xlabel(r'$\nu$ [$\mu$Hz]')  
    plt.ylabel(r'[ppm$^2$/$\mu$Hz]')
    plt.axis(ymin=psdme[-1]/100.,xmax=np.max(single_nu))
    plt.legend(loc=0)
    if(Pdf):
        fname = OutDir + StarName+ '_fig1.pdf'
    else:
        fname = OutDir + StarName+ '_fig1.png'
    plt.savefig(fname)

    plt.figure(101)
    plt.clf()
    plt.title(StarName+ ' Averaged LC')
    plt.plot(single_ts[:,0]/86400.,single_ts[:,1]*1e-4,'grey')
        
    m = int( round(max(3600.,Sampling)/Sampling))
    p = int(nt/m)
    tsm = rebin1d(single_ts[0:p*m,1],p)/float(m)
    timem = rebin1d(single_ts[0:p*m,0],p)/float(m)
    
        
    plt.plot(timem/86400.,tsm*1e-4,'k')
    plt.xlabel('Time [days]')  
    plt.ylabel('Relative flux variation [%]')
    if(Pdf):
        fname = OutDir + StarName+ '_fig5.pdf'
    else: 
        fname = OutDir + StarName+ '_fig5.png'
    plt.savefig(fname)


    if(ExtendedPlots):

        plt.figure(102)
        plt.clf()
        plt.title(StarName+ ' PSD (zoom)')
        numax =  opar[2]['numax']
        Hmax = opar[2]['Hmax']  
        u =     (num> numax*0.5) & (num < numax*1.5)
        plt.plot(num[u],psdm[u],'k',lw=2) # simulated spectrum, all camera
    
        plt.loglog()
        plt.xlabel(r'$\nu$ [$\mu$Hz]')  
        plt.ylabel(r'[ppm$^2$/$\mu$Hz]')    
        if(Pdf):
            fname = OutDir + StarName+ '_fig2.pdf'
        else:
            fname = OutDir + StarName+ '_fig2.png'
        plt.savefig(fname)
    
    
        plt.figure(103)
        plt.clf()
        plt.title(StarName+ ' PSD Stellar components')
        u =     (f> numax*0.5) & (f < numax*1.5)
        plt.plot(f[u], 0.5*mps_nf[u],'r',lw=2,label='star') # noise free
        plt.xlabel(r'$\nu$ [$\mu$Hz]')  
        plt.ylabel(r'[ppm$^2$/$\mu$Hz]')    
        if(Pdf):
            fname = OutDir + StarName+ '_fig4.pdf'
        else: 
            fname = OutDir + StarName+ '_fig4.png'
        plt.savefig(fname)

        if(SystematicsEnable):
            plt.figure(104)
            plt.clf()
            plt.title(StarName+': systematic LCs [ppm]')
            for i in range(NGroup):
                for j in range(NCamera):
                    plt.plot(full_ts[:,i,j,0]/86400.,full_ts_SC[:,i,j,1])
            plt.plot(single_ts[:,0]/86400.,single_ts_SC,'k',lw=2)
            plt.xlabel('Time [days]')
    
    plt.draw()
    plt.show()


if(Verbose):
    print ('done')
    
if(Verbose | Plot):
    s=input('type ENTER to finish')



