import numpy as np
import math
import re
import scipy.optimize
import FortranIO
import universal_pattern as UP
 
''''

Solar-like Light-curve Simulator (SLS)

Simulate stochastically-excited oscillations and associated stellar background (granulation and activity).

Two different types of oscillation spectrum can be simulated:
- gen_up : oscillation spectrum based on the Universal Pattern  by Mosser et al (2011, A&A, 525, L9).
   Mixed-modes can been included when the asymptotic period spacing is specified.

- gen_adipls: oscillation spectrum based on eigenfrequencies computed with ADIPLS 

For more details see the helps associated with the procedures gen_up() and gen_adipls()

Other procedures are:

- genslc():  from an individual simulation, generate a series of simuated ligth-curves and save them in a set of files (one per light-curve). Each simulated light curve corresponds to a given realization. 

- savelc() : save a given light-curve in a ASCII file

- test_gen_up() : perform a simple illustrative test assuming the Universal Pattern (call to  gen_up)

- test_gen_adipls() : perform a simple illustrative test  using a set of eigenfrequencies computed with ADIPLS  (call to gen_adipls)



Copyright (c) 2014, Reza Samadi, LESIA - Observatoire de Paris

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

# numax reference value (Mosser et al 2013, SF2A)
numaxref= 3104.

# solar constants
nucsun = 5300.  # sun cutt-off freq.
numaxsun = 3100.
deltanusun = 135.
teffsun = 5777.
msun = 1.98919e33           # solar mass
rsun = 6.9599e10            # solar radius
gmsun = 1.32712438e26       # G Msun
logg_sun = math.log10(gmsun) - 2.*math.log10(rsun)
ggrav = gmsun/msun          # grav. constant G

    
class SLSError(Exception):
    def __init__(self, value):
             self.value = value
    def __str__(self):
        return repr(self.value) 

def rebin1d(array,n):
    nr=int(float(array.shape[0])/float(n))
    return (np.reshape(array,(n,nr))).sum(1)

def gaussenvelop(nu,A,nu0,width):
    '''
    Gassian envelop for the oscillation spectrum

    INPUTS
    nu: frequencies
    A: amplitude of the envelope
    nu0: position in frequency (numax)
    width: width of the envelope
    
    '''
    return A*np.exp(-((nu-nu0)**2)/(width**2/(4*math.log(2)))) 
        
def plf(nu,h,tau,alpha):
    '''
    Pseudo Lorentzian Function (plf) for the granulation component(s)

    INPUTS:
    nu:  frequencies
    h: height
    tau : characteristic time
    alpha : slope
    
    '''

    return h/(1. + (2.*math.pi*tau *nu)**alpha)

def eft_2plf(h1,tau1,alpha1,h2,tau2,alpha2):
    # Given two pseudo lorenztian functions (plf), compute the associated e-folding time (eft)
    N = 2**19
    Nf = N/2  + 1
    tau = min(tau1,tau2)
    sampling =  tau*1e-3
    dnu = 1./(sampling*N)
    nu = np.arange(Nf) * dnu

    spc = plf(nu,h1,tau1,alpha1) + plf(nu,h2,tau2,alpha2)
    acf = np.fft.irfft(spc,n=N)
    acf = acf[1:N//2]/acf[0]
    time = (np.arange(N/2-1)+1.)*sampling/tau
    eft = np.interp(math.exp(-1),acf[::-1],time[::-1])*tau

    return eft

def mpsd2rts(f,mpsd,seed=None,time_shift=0.):
    '''

    Simulate a random time-series from a mean power spectral density spectrum

    time, ts, ps, seed =  mpsd2rts(f,mpsd,seed=None)

    INPUTS:
    f: frequencies (Hz)
    mpsd : mean power spectral density (PSD) in /Hz. Single-sided PSD
    seed: seed of the Random Number Generator (RNG)
    time_shift : imposes a time shift (seconds), default: 0.

    OUTPUTS:
    time: time values
    ts: simulated time-series
    ps: simulated power spectrum [/Hz]
    seed:  adopted value of the seed (chosen by the Random Number Generator if seed is not specified in input)
    
    '''

    # dnu: frequency resolution in Hz
    if (f[0] ==0. ):
        dnu = f[1]
        n = f.size
        i0 = 0
    else:
        dnu = f[0]
        n = f.size + 1
        i0 = 1

    # sampling time [s]
    dt = 1./(2.*dnu*n)

    # 
    time = np.arange(2*n) * dt  + time_shift

    if( seed == None):
        # see is not specified, we choose a random value 
        seed = np.random.random_integers(0,high=2**30)    
    # initialization of the state of the RNG
    np.random.seed(seed)
    
    
    # real and imaginary part randomly distributed
    Rr = np.random.normal(loc=0.,scale=1./math.sqrt(2.),size=n-i0)
    Ri = np.random.normal(loc=0.,scale=1./math.sqrt(2.),size=n-i0)


    # fourier spectrum
    tf = np.zeros(n,dtype='complex')
    
    tf[i0:] = np.sqrt(mpsd) * (Rr + Ri*1j)  * np.sqrt(n/dt)
    if(time_shift !=0):
        fp = np.arange(n)*dnu # frequencies, Hz
        tf[i0:] *= np.exp(2.*math.pi*fp[i0:]*time_shift*1j)

    # symmetrise
    tf = np.hstack((
                    np.array([0]),
                    tf[1:],
                    np.array([0]),
                    tf[:0:-1].conj()))        


    # generate the time series [ppm]
    ts = np.real(np.fft.ifft(tf)) 

    # power spectrum in /Hz
    ps = np.abs(tf[i0:n])**2 *(dt/n)

    return time,ts,ps,seed

def power_ratio(l,beta):
    '''

    power_ratio (l,i)

    Returns the mode relative power amplitude for a given angle i with respect to the mode power amplitude at the angle i=0, 
    i.e. returns the quantity given by (see eg. Dziembowski, 1971, Toutain & Goutterbroze 1993) :
    Q^2_lm = (l-m)!/(l+m)! ( P_lm[cos(i)] )^2
    where i is the inclination angle (i=0 corresponds to the direction of the rotation axis, ie. theta=0).  
        
    INPUTS:
    l: mode angular degree
    i: inclination angle in degree (i=0 points toward the observer)
    
    '''

    def combi (n,r):
        '''
        Compute combinations n r,  for  n>r

        '''
        c = math.factorial(n)/math.factorial(n-r)/math.factorial(r)
        return c



    def dmm(l,m1,m2,beta):
        '''

        compute dm1,m2(beta)  for  m1+m2>=0  and m1-m2>=0

        '''
        co = math.cos(beta/2.)
        si = math.sin(beta/2.)

        sum=0.
        for s in range(0,l-m1+1):
            var=0.
            var=combi(l+m2,l-m1-s)*combi(l-m2,s)*(-1.)**(l-m1-s)
            var=var*co**(2*s+m1+m2)*si**(2*l-2*s-m1-m2)
            sum=sum+var

        sum=sum*math.sqrt(math.factorial(l+m1)*math.factorial(l-m1)*1.)
        sum=sum/math.sqrt(math.factorial(l+m2)*math.factorial(l-m2)*1.)

        return sum


    def function_rot(l,beta):

        dim = 2*l+1
        mat = np.zeros((dim,dim))
        for i in range(0,l+1):
            for j in range(-i,i+1):
                  mat[i+l,j+l] = dmm(l,i,j,beta)

        for i in range(-l,1):
              for j in range(i,-i+1):
                  mat[i+l,j+l] = mat[-i+l,-j+l]*(-1.)**(i-j)

        for j in range(0,l+1):
            for i in range(-j,j+1):
                  mat [i+l,j+l] = dmm(l,j,i,-beta)

        for j in range(-l,1):
            for i in range(j,-j+1):
                  mat [i+l,j+l] = mat[-i+l,-j+l]*(-1.)**(i-j)

        return mat

    angle = math.pi*beta/180.
    zz = function_rot(l,angle)
    zz = zz[:,l]**2
    norm = zz.sum()
    return zz/norm

def activity_spectrum(f,parameters,verbose=0):
    """
    f : frequency values [muHz]
    parameters:  (sigma,tau) where  sigma: amplitude (in ppm) and tau: characteristic time (in days)
    An "single-sided power spectrum" is assumed

    """
    sigma= parameters[0] # in ppm
    tau =parameters[1]*86400. # characteristic time, converted from days to seconds
    H = 4.*tau*sigma**2*1e-6
    if(verbose):
        print (("activity parameters: sigma=%f [ppm], tau= %f [days] %f [s]") % (sigma,tau/86400.,tau))
    return plf(f,H,tau*1e-6,2.)

def granulation_spectrum(f,numax,verbose=0,type=0,mass=1.,teff=-1.,radius=-1.):
    
    '''
   Granulation spectrum
   
   GS,parameters = granulation_spectrum(numax,tau)
   We assume an "single-sided power spectrum"

   INPUTS:
   f : frequency values [muHz]
   numax: peak frequency [muHz]
   type :
   =0 -> assume standard scaling relations and a Lorentzian form
   =1 -> as in Kallinger et al (2014)
   mass : star mass (used for type=1)
   
   OUTPUTS:
   GS: the granulation spectrum computed for the frequency set f
   parameters: granulation parameters, the returned parameters depend on "type"

   type = 0 : parameters = (H,tau,sigma)
      H: Height of the granulation component [ppm2/muHz]
      tau: time-scale [s]
      sigma: amplitude of the granulation [ppm]

   type = 1 : parameters = (h1,tau1,c1,h2,tau2,c2)
   
    '''
    if (type==0):
        
        tau_gran =  9e4*(numax/10.)**(-0.90) # time scale [seconds]  ref. M13
        taue = tau_gran # e-folding type, taue = tau_gran for a Lorentzian function
        #    h_gran =  2e6*(tau_gran/1e5) **2.34 # height [ppm2/muHz]  ref.  M13
        h_gran = 4e5*(numax/10.)**(-2.15) # height [ppm2/muHz]  ref.  M13
        sigma_gran =  math.sqrt(h_gran*1e6/tau_gran/4.)
        if(verbose):
            print (("tau [s], H [ppm2/muHz], sigma [ppm]: %e, %e, %f, %g")%   (h_gran , tau_gran , sigma_gran,taue))
        gran_par = {"h1": h_gran, "tau1": tau_gran, "alpha1": 2.,"taue": taue}
        return plf(f,h_gran,tau_gran*1e-6,2.), gran_par
    else:
##        if (True):
        if ( (numax < 2.) | (numax > 3500.)):
            print ("Outside the valid numax domain of Kallinger et al (2014) scaling relations for the granulation spectrum")
            print ("Assuming Samadi et al (2013)'s scaling relations")
            if( (radius <0) | (mass <0.) | (teff<1)):
                raise SLSError("mass, teff and radius must be specified")
            Ma0 = 0.258
            taue_sun = 1.72e2 * 1e-6 # / 10^6 s
            logg = math.log10(mass/radius**2) + logg_sun
            Ma = Ma0 * (teff/teffsun)**2.35 * (10.**(logg -logg_sun))**(-0.152)
            taue = taue_sun*(numaxref/numax)*(Ma0/Ma) 
            fg = -0.67 + 2.3*(Ma/Ma0)  -0.59*(Ma/Ma0)**2
            sigsun = 38.6	
            sig2 = sigsun**2 * (teff/teffsun)**(3./2.)*(1./mass)*(numaxref/numax)*fg**4  # Eq. 14
            Cbol = (teff/teffsun)**0.8
            h = sig2*taue*4./Cbol**2
            if(verbose):
                print (("Ma, fg, sig2, taue; Cbol, H : %f, %f, %f, %f, %f, %f") % (Ma,fg,sig2,taue,Cbol,h))
            
            gran_par = {"h": h, "tau": taue, "alpha": 2,"taue": taue}
            
            return  plf(f,h,taue,2) , gran_par

        a1k = 3382
        a1s = -0.609
        a1t = 0.

        a2k = 3710.
        a2s = -0.613
        a2t = -0.26

        b1k = 0.317
        b1s = 0.970

        b2k = 0.948
        b2s = 0.992

        c1 = 4.
        c2 = 4.
	
        a1 = a1k*numax**a1s*mass**a1t
        a2 = a2k*numax**a2s*mass**a2t

        b1 = b1k*numax**b1s
        b2 = b2k*numax**b2s


        xi1 = 2.*math.sqrt(2.)/math.pi
        tau1   = 1./(2.*math.pi)/b1

        xi2 = xi1
        tau2   = 1./(2.*math.pi)/b2

        h1 = a1**2/b1*xi1
        h2 = a2**2/b2*xi2
        taue = eft_2plf(h1,tau1,c1,h2,tau2,c2) # e-folding time
        gran_par = {"h1": h1, "tau1": tau1, "alpha1": c1,"h2": h2, "tau2": tau2, "alpha2": c2,"taue": taue}
        if(verbose):
            print (("h1 [ppm2/muHz], tau1 [s], c1, h2 , tau2, taue , c2: %g, %g, %g, %g, %g, %g, %g") % (h1,tau1*1e6,c1,h2,tau2*1e6,c2,taue*1e6))

        return  plf(f,h1,tau1,c1) + plf(f,h2,tau2,c2),gran_par

def amax(numax,deltanu,teff):
    '''

    Scaling relation for the oscillation maximum (Amax) as established by Corsaro et al (2013, MNRAS 430, 2313-2326)

    We assume the model named  4\beta (see Table 3 & 4)
    
    If numax >150 muHz,  coefficients established using the short cadence data (table 3) are assumed otherwise those obtained with the long cadence data (table 4).
    
    
    Return the value of Amax bolometric (ppm) as a function of numax (muHz), deltanu (muHz) and teff (K)
    

    '''

    Amaxbolsun = 2.53 # rms value (see Michel et al 2009)


    if ( (numax > 4000.) | (numax < 2.) ):
        print ("WARNING ! numax outside the domain for the oscillation maximum scaling relation of  Corsaro et al (2013), assuming Samadi et al (2007)'scaling relation")
        Amaxbol = Amaxbolsun * (teff/teffsun)**(7./2.)*(numaxsun/numax)
        return Amaxbol

    if( numax > 150.):
        # parameters of model 4\beta for short cadence data
        s = 0.748
        r = 3.47
        t = 1.27
        lnbeta = 0.321

        ## model M1 
        ## s = 0.624
        ## r = 3.68
        ## lnAmax = -s*math.log(numax/numaxsun) + (3.5*s -r + 1.) *math.log(deltanu/deltanusun)
        ## Amaxbol = math.exp(lnAmax) * (teff/5934.)**(0.8) *Amaxbolsun
    else:
        # parameters of model 4\beta for long cadence data
        s = 0.602
        r = 5.87
        t = 1.32
        lnbeta = 0.45

    lnAmaxbol = (2.*s-3.*t)*math.log(numax/numaxsun) + (4.*t-4.*s)*math.log(deltanu/deltanusun) + (5.*s-1.5*t-r+0.2)*math.log(teff/teffsun) + lnbeta

    Amaxbol = math.exp(lnAmaxbol)*Amaxbolsun

    return Amaxbol


def lwmax(teff,numax):
    '''
    Scaling relation for the mode linewidth at numax as established by Appourchaux et al (2012, A&A, 537, A134)

    Return the mode linewidth (muHz) at numax as a function of teff (K)
    
    '''
    if ( (teff <= 6700.) & (teff >= 5300.) ):
        # Appourchaux et al (2012), table 2, at maximum mode height
        Gamma0 = 0.20
        alpha = 0.97
        s = 13.0
        return  Gamma0 + alpha*(teff/teffsun)**s
    else:
        if( (teff<4000.) | (teff>6700.)):
            print("Warning !! Teff outside the domain for the mode line-width scaling relations")
##    elif( (teff<=5300.) & (teff>=4000.)):
        # Belkacem, 2012, A&A
        numax0 = 909.253761301
        return  0.19*(teff/4800.)**(10.8-0.3*0.5)*(numax/numax0)**(-0.3) #
##        return  0.19*(teff/4800.)**(10.8)
    
        
    

def get_obs_par(mag, T, dt , pn_ref = 7.7, mag_ref = 6.,wn_ref = 0. , verbose = False, teff = 4800.):
    
    # total white-noise level ppm.s^(1/2)
    W = math.sqrt ( pn_ref**2 * 3600.*10.**((mag-mag_ref)/2.5)  + wn_ref**2    * 3600.*10.**(2.*(mag-mag_ref)/2.5) )
    if(verbose):
        print ('Total white-noise [ppm.Hz^(-1/2)]: %e' % W)
        print ('Total white-noise at sampling time: %e' % (W*math.sqrt(1./dt)))
 ## 2.*W**2*1e-6

    T0 = T*86400. # duration in seconds
    
    # size of the time series
    n = int(T0/(dt))

    # frequency resolution [muHz]
    dnu = 1./(n*dt)*1e6


    nyq = 1./(2.*dt)*1e6 # Nyquist frequency [muHz]

    m2b = (teff/5934.)**(0.8) # conversion from  measured to bolometric amplitude (Ballot et al 2011)

    # observation parameters
    obs_par = {'magnitude': mag, 'duration': T, 'sampling': dt, 'frequency_resolution': dnu, 'nyquist': nyq, 'white_noise': W, 'time_steps_number': n , 'bolometric_coefficient': m2b}
    
    return obs_par


def read_agsm(file):
    '''
     modes,starpar,iekinr = read_agsm(file)

    
     OUTPUTS:
     modes : mode parameters:
     
     modes[:,0] : n order
     modes[:,1] : l degree
     modes[:,2] : nu frequency in muHz (eigenvalue)
     modes[:,3] : square normalised frequency (w2)
     modes[:,4] : Richardson frequency in muHz
     modes[:,5] : variationnal frequency in muHz
     modes[:,6] : icase
     modes[:,7] : mode inertia g cm^2  x 10^40
     modes[:,8] : beta = (1  - Cnl ) where cnl is the Ledoux coefficient,  available if irotkr is set to 1.
     modes[:,9] : zeta = Ecore/E  available if irotkr is set to 1 otherwise zeta = 0
     starpar : star parameters (mass,radius)
     iekinr: ADIPLS option
     
    '''

    f = FortranIO.FortranBinaryFile(file, mode='rb')

    cs = []
    while True:
        try:
            csi = f.readRecordNative('d')
            cs.append(csi)
        except IndexError:
            break

    nmodes = len(cs)
    modes = np.zeros((nmodes,10))
    mstar = (cs[0])[1]
    rstar = (cs[0])[2]
    for i in range(nmodes):
        modes[i,0] = (cs[i]) [18] # n
        modes[i,1] = (cs[i]) [17] # l        
        sigma2 = (cs[i]) [19]
        omega2 = sigma2*ggrav*mstar/(rstar**3)
        modes[i,2] = math.sqrt(omega2)/(2.*math.pi)*1e6 # nu in muHz
        modes[i,3] = sigma2
        modes[i,4] = (cs[i]) [36]*1e3 # Richardson frequency in muHz
        modes[i,5] = (cs[i]) [26]*1e3 # variational frequency in muHz
        modes[i,7] =  (cs[i]) [23] # * (mstar*1e-20) * ( (cs[i])[29 ] *  rstar*1e-10  )**2 *  4. *math.pi 
        modes[i,8] = (cs[i]) [35] # beta_nl   available if irotkr is set to 1.
        modes[i,9] = (cs[i]) [37] # zeta = Ecore/E  available if irotkr is set to 1.

    f.close()
    # reading separately the 'ics' integer variables
    f = FortranIO.FortranBinaryFile(file, mode='rb')
    n = 38*2
    iekinr = False
    for i in range(nmodes):
        data = f.readRecordNative('i')
        ics = data[n:n+8]
        iekinr_i = ics[6]
        iekinr |= iekinr_i
        # print(f'iekinr = {ics[6]}')
        # print(f'iekinr = {ics[6]}')
        modes[i,6] = ics[4]

    f.close()
    return modes,(mstar/msun,rstar/rsun),iekinr

def osc_spectrum_text_file(starID,f,modes,verbose=False,plot=False,win=0):

    spec = np.zeros(len(f))
    df = f[1] - f[0]
    nm = modes.shape[0] #modes: n, l, m , nu, gamma, h
    imax = np.argmax(modes[:,5])
    Gamma = modes[imax,4]
    Hmax = modes[imax,5]
    Amax = np.sqrt(Hmax*math.pi*Gamma/2.) # Eq. 12
    width_env = -1. # not relevant in this context
    for i in range(nm): # loop over the modes (n,l)
        # n = int(modes[i,0])
        # ell = int(modes[i,1])
        # m = int(modes[i,2])
        nu = modes[i,3] #  eigen frequency i
        gamma =  modes[i,4]
        h = modes[i,5]
        if ( gamma > df*2.): # resolved profile
            P =  1./ ( 1. + ( 2.*(f-nu)/gamma)**2 )   # Lorentzian : modes shape
        else: # un-resolved profile
            P = (math.pi*gamma/2./df)*( np.sinc( math.pi*(f-nu)/df) )**2
        spec[:] = spec[:] + P * h


    if(plot):
        import matplotlib
##        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        starName = ("%10.10i") % starID

        plt.close('all')

        plt.figure(win)
        plt.clf()
        plt.plot(modes[:,2],modes[:,3],'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$\Gamma$ [$\mu$Hz]')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )

        win += 1
        plt.figure(win)
        plt.clf()
        plt.plot(modes[:,2],modes[:,3],'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$H$ [ppm$^2/\mu$Hz] ')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )


        plt.show(block=False)
        ## s=raw_input('Type ENTER to continue')
    return spec,(Amax,Hmax,Gamma,width_env),None


def osc_spectrum_model_adipls(starID, f,modes,numax,deltanu,teff,verbose=False,a=0.,b=1.,plot=False,win=0,
                              rot_period_sur = 0., incl = 0., path='./', rot_core_f = 0.):
    '''

    Derive the oscillation spectrum from the theoretical mode properties (frequency and inertia)

    # a,b  surface effect parameters (Lorentzian function)
    # rot_period_sur : period of the surface rotation, in days. Do not include splittings if rot_period_sur  < 0

    '''

    starName = ("%10.10i") % starID
    Gamma =lwmax(teff,numax)
    Amax = amax(numax,deltanu,teff)
    Hmax = 2.*Amax**2/(math.pi*Gamma) # Mode height at numax assuming resolved modes (from Lochard et al 2005, A&A, 438, 939, eq. 16)
    # the additional factor comes from the fact that Lochard et al 2005 assumed two-side power density spectrum
    # here we assume an single-sided power density spectrum
    nm = modes.shape[0]
    width = 0.66*(numax**0.88)        # dnu_env = FWHM of the excess power envelope, see: Mosser et al. (2012 A&A 537 A30) - Table 2

    if(verbose):
        print ('------------------------------------------------------------------------')
        print(('Amax=',Amax,' [ppm]'))
        print(('Gamma=',Gamma, ' [muHz]'))
        print(('Hmax=',Hmax,' [ppm^2/muHz]'))
        print(('Width=',Hmax,' [muHz]'))
        print ('theoretical input frequencies:')
        print( "n, l, nu_eig, nu_var , nu_eig-nu_var,nu_eig-nu_richardson inertia")
        u = modes[:,6] == 10010
        for i in range(nm):
            if (u[i]):
                print (("%i %i %f %f %f %f %e") % (modes[i,0],modes[i,1],modes[i,2],modes[i,5],modes[i,2]-modes[i,5],modes[i,2]-modes[i,4],modes[i,7]))
        print ('------------------------------------------------------------------------')
    spec = np.zeros(len(f))
    df = f[1] - f[0]

    # square visibility
    V1 = 1.5
###    V1 = 0.2
    V2 = 0.5
    V3 = 0.05
    V = (1.,V1,V2,V3)

    # identification of the radial modes 
    ir = ( modes[:,1] ==0)
    nr = ir.sum()
    Imax = np.interp(numax,modes[ir,2],modes[ir,7])
    NU = []
    L = [] # angular degrees
    LM = []  # azimuth degree
    N = []
    G = []
    H = []
    I = []
    DNU = [] 
    gauss = lambda nu : np.exp(-((nu-numax)**2)/((2.*width)**2/(4*math.log(2))))
    ## A = 2.
    Ap = 2.
    An = 6.
    fout = open(path+starName +  '.modes','w')
    fout.write("# n, l, m , nu, gamma, h, I/Imax, dnu, splitting\n")
    if (rot_period_sur > 0):
        rot_env_f = 1. / (rot_period_sur * 86400.) * 1e6
    else:
        rot_env_f = 0.
    if(verbose):
        print(f'rot_env_f = {rot_env_f} muHz')
        print(f'rot_core_f = {rot_core_f} muHz')

    for i in range(nm): # loop over the modes (n,l)
        inertia = modes[i,7]
        nu0 = modes[i,2] #  eigen frequency i
        n = int(modes[i,0])
        ell = int(modes[i,1])
        if (rot_env_f> 0 or rot_core_f>0.):
                nlm = 2*ell + 1 # total number of the multiplets
                lm1 = -ell
                lm2 = ell
        else:
                nlm = 1
                lm1 = 0
                lm2 = 0
        if (rot_env_f>0. or rot_core_f>0.):
            pr = power_ratio(ell,incl)
        else:
            pr = np.ones(nlm)
        beta = modes[i, 8]
        if(ell>0 and  beta==0.): # missing information about the splitting, assuming beta = 1
            rot_env_f_beta = rot_env_f
        else:
            rot_env_f_beta = rot_env_f * beta
        for lm in range(lm1,lm2+1): # loop over the multiplets
            # computing the splitting
            if(ell==1): # for dipolar modes
                split = lm*0.5 * (modes[i,9] * (rot_core_f - 2. * rot_env_f_beta) + 2. * rot_env_f_beta)  # Eq. 21  & 22 in Goupil et al 2013
            else: # for l>1
                split = lm*rot_env_f_beta
            # print(f'i = {i}  l = {ell} m = {lm} zeta = {modes[i, 9]}  split = {split}')
            icase = int(modes[i,6])
    ##        dnu = a*numax*(nu/numax)**b # Kjeldsen & Bedding 2008
    ##        dnu = numax * (c + a/(1. + (nu/numax)**b) )
            nu = nu0 + split
            dnu = a* numax * (1. -  1./(1. + (nu/numax)**b) ) # surface effects, Eq. 9, Sonoi et al 2015, A&A, 583, 112
            nu = nu + dnu  # we add surface effects
            h = gaussenvelop(nu,Hmax,numax,width)*V[ell]*pr[lm-lm1]
##            gamma = Gamma*(Imax/inertia)* (A - (A-1.)*gauss(nu))
            gamma = Gamma*(Imax/inertia)* (1. + (Ap*(nu>=numax) + An*(nu<numax) )* (1. - gauss(nu)))
            if (icase != 10010):
##                if(verbose):
##                    print 'excluded mode according to icase=',icase
                continue
            '''
            if (n  <= 0):
##                if(verbose):
##                    print 'excluded mode because n<=0 : ',n
                continue
            '''
            if (ell  > 3):
##                if(verbose):
##                    print 'excluded mode because l>3 : ',ell
                continue
            fout.write( ("%i %i %i %f %e %e %e %f %f\n") % (n,ell,lm,nu,gamma,h,inertia/Imax,dnu,split))
            if ( gamma > df*2.): # resolved profile
                P =  1./ ( 1. + ( 2.*(f-nu)/gamma)**2 )   # Lorentzian : modes shape
            else: # un-resolved profile
                P = (math.pi*gamma/2./df)*( np.sinc( math.pi*(f-nu)/df) )**2
            spec[:] = spec[:] + P * h
            NU.append(nu)
            L.append(ell)
            LM.append(lm)
            N.append(n)
            G.append(gamma)
            H.append(h)
            I.append(inertia/Imax)
            DNU.append(dnu)
        
    N = np.array(N)
    L = np.array(L)
    NU = np.array(NU)
    G = np.array(G)
    H = np.array(H)
    I = np.array(I)
    DNU =  np.array(DNU)
    if(plot):
        import matplotlib
##        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.close('all')
        plt.figure(win)
        plt.clf()
        plt.plot(NU,G*I,'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$\Gamma (I/I_{\rm max})$ [$\mu$Hz]')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )

        win += 1
        plt.figure(win)
        plt.clf()
        plt.plot(NU,G,'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$\Gamma$ [$\mu$Hz]')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )

        win += 1
        plt.figure(win)
        plt.clf()
        plt.plot(NU,I,'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$I/I_{\rm max}$')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )

        win += 1
        plt.figure(win)
        plt.clf()
        plt.plot(NU,H,'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$H$ [ppm$^2/\mu$Hz] ')
        plt.semilogy()
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )

        win += 1
        plt.figure(win)
        plt.clf()
        plt.plot(NU,DNU,'ro')
        plt.xlabel(r'$\nu$ [$\mu$Hz]')
        plt.ylabel(r'$\delta \nu$ [$\mu$Hz]')
        plt.title(starName)
        plt.savefig(starName +  ("_%2.2i.pdf") % (win) )
    
        plt.show(block=False)
        ## s=raw_input('Type ENTER to continue')
    return spec,(Amax,Hmax,Gamma,width),(N,L,LM,NU,G,H,I,DNU)

def gen_osc_spectrum(starID, filename, teff,mass,radius,dt,T, mag , seed = None, pn_ref = 7.7, wn_ref = 0. ,
                     mag_ref = 6.,verbose = False, GST = 1 , a=0 , b=1.  ,  plot=False, rot_period_sur = 0.,
                     incl = 0., activity = None, granulation = True , path= './' , oscillation=True,
                     type = 1, numax = -1., deltanu = -1., rot_core_f = 0.) :
    '''

    (time,lc,nu,pds,mpds,par,seed) = gen_osc_spectrum( starID, filename, teff, dt,T, mag , seed = None, pn_ref = 7.7, wn_ref = 0. , mag_ref = 6., verbose = False, GST = 1 , a=0 , b=1.  ,  plot=False)
    
    Generate a simulated light-curve using eigenfrencies computed with ADIPLS or coming from an input text file

    INPUTS:

    starID : any number to identify the simulation, this is used only for the plots and the generated output files 
    filename: name of the input file providing the mode properties (the format of this file depends on the parameter type, see below)
    teff: effective temperature [K]
    dt: sampling time [s]
    T: simulation duration [days]
    mag: magnitude of the star
    seed: seed of the Random Number Generator
    pn_ref: reference value of the photon noise [in ppm per hour]
    wn_ref: reference value of the non-photonic white noise  [in ppm per hour]
    mag_ref: reference magnitude for which pn_ref and wn_ref are specified

    GST: type of granulation spectrum (see the function named 'granulation_spectrum')
    a,b,c: coefficients involved in the formulation used of the surface correction
    plot: if True  do some plot
    verbose: if True be verbose
    rot_period_sur : period of the surface rotation, in days. Do not include splittings if rot_period_sur  < 0
    incl : inclination angle (in degrees)

    activity: add an activity component, tuple (sigma,tau) where  sigma is the amplitude (in ppm) and  tau the characteristic time (in days) ; activity= None  if no activity is added (default case)
    granulation: add the granulation component (by default: True)
    oscillation: turn on oscillation (default: True)    
    type: type of the oscillation spectrum. Default: type = 1
        type = 0 : the mode properties (freq., width and height) are read from an input text file
        type = 1:  the mode properties are derived from an ADIPLS file (.gsm file)

    path: where to save the information about the modes

    OUTPUTS:
    time : time vector (in seconds)
    lc: ligth-curve
    nu: frequencies [muHz]
    pds: power density spectrum [ppm^2/muHz]
    mpds: mean power density spectrum  [ppm^2/muHz]
    par: adopted paramaters of the simulation
    seed: adopted value of the seed
    
    '''
    
    obs_par =  get_obs_par(mag, T, dt , pn_ref = pn_ref,  wn_ref = wn_ref, mag_ref = mag_ref,verbose = verbose, teff=teff)

    dnu = obs_par['frequency_resolution']
    n = obs_par['time_steps_number']
    W = obs_par['white_noise']
    nyq =  obs_par['nyquist']
    
    # frequency values [muHz], n-1 values, null frequency is excluded
    f = (np.arange(n/2-1)+1)*dnu
    b2m = 1./ obs_par['bolometric_coefficient'] # bolometric to measured amplitude coefficient
    if(verbose):
        print('mode properties load from %s' % filename)
    if(type==0): # text file
        modes_tmp = np.loadtxt(filename) # (nu, gamma, h)
        modes = np.zeros((modes_tmp.shape[0],6))
        modes[:,3:6] = modes_tmp
        imax = np.argmax(modes[:,5])
        if(numax<0):
            numax = modes[imax,3]
        if(deltanu<0):
            deltanu = modes[-1,3] - modes[-2,3]
        mass = (numax/numaxref)**3 * (deltanu/deltanusun)**(-4)*(teff/teffsun)**(3./2.)
        radius = (numax/numaxref) *  (deltanu/deltanusun)**(-2)*(teff/teffsun)**(1./2.)
        star_par = {'mass': mass, 'teff' : teff, 'radius' : radius, 'logg': math.log10(mass/radius**2) +logg_sun}
        # if (rot_period_sur > 0 or rot_core_f>0. ):  # mode splitting cannot be included
        #     raise SLSError("cannot include mode splitting, turn-off surface and core rotation")
        OS,(Amax,Hmax,Gamma,width_env),modespar = osc_spectrum_text_file(starID,f,modes,verbose=verbose,plot=plot)
    elif(type==1): # .gsm file
        modes, (mass, radius),iekinr = read_agsm(filename)
        star_par = {'mass': mass, 'teff' : teff, 'radius' : radius, 'logg': math.log10(mass/radius**2) +logg_sun}
        numax =  numaxref * (mass/radius**2) * math.sqrt(teffsun/teff)
    ##    deltanu = deltanusun * (mass/radius**3)
        deltanu = deltanusun * math.sqrt(mass/radius**3)
        if(verbose):
            print (('numax = %f, deltanu = %f [muHz]') % (numax,deltanu))
        if (rot_core_f>0.):  # mode splitting for dipolar, we check if the zeta coefficients is
             if (not iekinr):
                 raise SLSError("no zeta coefficients in the GSM file, mode splitting cannot be included for dipolar mode. Change .gsm file or turn-off core rotation")
        OS,(Amax,Hmax,Gamma,width_env),modespar = osc_spectrum_model_adipls(starID,f,modes,numax,deltanu,teff,
                                                                            verbose=verbose,a=a,b=b, plot=plot,
                                                                            rot_period_sur = rot_period_sur , incl = incl,
                                                                            path = path,rot_core_f=rot_core_f)
    elif(type==2): # from the grid data set
        nmodes = len(filename['freq'])
        modes = np.zeros((nmodes, 10))
        if(verbose): print('number of mode eigenvalues: %i' % nmodes)
        modes[:,0] = filename['n']
        modes[:,1] = filename['l']
        modes[:,2] = filename['freq']
        modes[:,6] = 10010 # icase
        modes[:,7] = filename['inertia']
        star_par = {'mass': mass, 'teff' : teff, 'radius' : radius, 'logg': math.log10(mass/radius**2) +logg_sun}
        numax =  numaxref * (mass/radius**2) * math.sqrt(teffsun/teff)
    ##    deltanu = deltanusun * (mass/radius**3)
        deltanu = deltanusun * math.sqrt(mass/radius**3)
        if(verbose):
            print (('numax = %f, deltanu = %f [muHz]') % (numax,deltanu))
        # if (rot_period_sur > 0 or rot_core_f>0.):  # mode splitting cannot be included
        #     raise SLSError("cannot include mode splitting, turn-off surface and core rotation")
        OS,(Amax,Hmax,Gamma,width_env),modespar = osc_spectrum_model_adipls(starID,f,modes,numax,deltanu,teff,verbose=verbose,a=a,b=b, plot=plot, rot_period_sur = rot_period_sur , incl = incl , path = path )

    else:
        raise SLSError("unknown type of oscillation spectrum")



    # Granulation spectrum (GS)
    if(granulation):
        GS, gran_par   = granulation_spectrum(f,numax,mass=mass,teff=teff,radius=radius,type=GST)
        GSf,  gran_par  = granulation_spectrum(2.*nyq-f,numax,mass=mass,teff=teff,radius=radius,type=GST,verbose=verbose)  # Folded component of the GS
    else:
        GS,GSf = 0.,0.
        gran_par = {}

    if activity is None:
        AS = 0.
        ASf = 0.
        activity_par = {'sigma_a': 0., 'tau_a': 0.}
    else:
        AS = activity_spectrum(f,activity,verbose=verbose)
        ASf = activity_spectrum(2.*nyq-f,activity)
        activity_par = {'sigma_a': activity[0], 'tau_a': activity[1]}

    # mean power spectrum [ppm2/Hz]
    SC = 1e6 *(np.sinc(0.5*f/nyq))**2
    mpsd_osc = OS * (b2m**2) * SC
    mpsd_gran = (GS  + GSf) * SC
    mpsd_act = (AS + ASf) * SC
    mpsd = 2.*W**2
    if(oscillation):
        mpsd += mpsd_osc
    if(granulation):
        mpsd += mpsd_gran
    if(activity is not None):
        mpsd += mpsd_act
    time, ts , ps , seed =  mpsd2rts(f*1e-6,mpsd,seed=seed)

##    osc_par = (numax,deltanu,Amax,Hmax,Gamma,width_env,a,b,c)
    if(oscillation):
        osc_par = {'numax' : numax, 'deltanu': deltanu, 'Amax' : Amax, 'Hmax' : Hmax, 'Gammamax': Gamma, 'width_envelop' : width_env, 'se_a': a, 'se_b': b }
    else:
        osc_par = None
        star_par = None

    if(verbose):
        print (("numax [muHz], delta_nu [muHz], mass, radius, Teff [K]: %f , %f, %f, %f, %f") % (numax,deltanu,mass,radius,teff))
        print (("Hmax [ppm2/muHz], Amax [ppm], gamma [muHz], width_env [muHz]: %e , %f , %f, %f") % (Hmax,Amax,Gamma,width_env))
        print (("a= %f, b= %f") % (a,b))
        print (("b2m: %f")% ( b2m ))
#        print (("b2m, white_noise  [ppm2/muHz]: %f %g")% ( b2m,2.*W**2*1e-6 ))
#        print (("times series, standard deviation [ppm]: %f") % np.std(ts))
#        print (("theoretically expected standard deviation [ppm]: %f") %  math.sqrt(  np.sum(mpsd)*1e-6* f[0] ))

    # output parameters
    opar = (star_par,obs_par,osc_par,gran_par,activity_par)

    return time,ts,f,ps*1e-6,mpsd*1e-6,mpsd_osc*1e-6,mpsd_gran*1e-6,mpsd_act*1e-6,opar,seed


def gen_up(starID, numax,dt,T, mag , delta_nu =  -1. , mass = -1. , seed = None, pn_ref = 7.7, wn_ref= 0.,
           mag_ref = 6., verbose = False, teff = 4800., DPI = -1.,  q = 0. , GST = 0 , incl = -1 , rot_core_f = 0., path= '',
           granulation = True , oscillation=True,activity=None, rot_period_sur = 0.):
    '''
    Generate a simulated light-curve on the basis of the Universal Pattern (UP)
    
    time,ts,f,ps,mps,opar,seed = gen_up(numax,dt,T, mag, delta_nu =  -1. ,teff = 4800., DPI = -1.,  q = 0.,pn_ref = 7.7,mag_ref = 6., seed = None,verbose= False)
    
    INPUTS:

    starID : any number to identify the simulation, this is used only for the plots and the generated output files 
    numax : peak frequency [muHz]
    dt : sampling time [s]
    T:  duration of the time series [days]
    mag: star magnitude

    delta_nu :  mean large separtion [muHz], if negative, delta_nu is computed according to a scaling relation
    teff : effective temperature [K], default: 4800 K
    DPI : asymptotic period spacing [s], negative if no mixed modes are wanted
    q : coupling coefficient, default: q=0 (no coupling)

    pn_ref: photon-noise reference level  [ppm per hour]
    wn_ref: Non photonic white-noise reference level  [ppm per hour]
    mag_ref: reference magnitude for the white-noise levels
    seed: seed of the Random Number Generator (RNG)
    verbose : verbose mode if True


    OUTPUTS:
    time: the time values
    ts: the simulated time-series (flux fluctuations in ppm)
    f: frequency values [muHz]
    ps: simulated power spectrum [ppm2/muHz]
    mps: mean power spectrum [ppm2/muHz]
    opar: simulation parameters
    seed:  adopted value of the seed (chosen by the Random Number Generator if seed is not specified in input)

    Simulation (output) parameters:
    - H_env : height of the oscillation envelope  [ppm^2/muHz]
    - width_env : width of the envelope [muHz]
    - gamma : mode width (for pure acoutic modes) [muHz]
    - h_gran : height of the granulation  component
    - tau_gran : time-scale of the granulation
    - sigma_gran : rms brightness fluctuations associated with the granulation [ppm]
    - b2m : bolometric to measured flux conversion factor
    parameters = (numax,delta_nu,mass,teff,H_env,width_env,gran_par,b2m,wn)    

    '''

    obs_par =  get_obs_par(mag, T, dt , pn_ref = pn_ref,  wn_ref = wn_ref, mag_ref = mag_ref,verbose = verbose, teff=teff)

    dnu = obs_par['frequency_resolution']
    n = obs_par['time_steps_number']
    W = obs_par['white_noise']
    nyq =  obs_par['nyquist']

    # frequency values [muHz], n-1 values, null frequency is excluded
    f = (np.arange(n-1)+1)*dnu
    f = (np.arange(n/2-1)+1)*dnu
    b2m = 1./ obs_par['bolometric_coefficient'] # bolometric to measured amplitude coefficient
    
    # Mean large separation
    if (delta_nu <=0. ):
        delta_nu = 0.274*numax**0.757 # ref. M13

    if (mass <=0.):
        # mass estimated combining the "classical" scaling laws for numax et deltanu
        mass = (numax/numaxref)**3. * (delta_nu/deltanusun)**(-4.) * (teff/teffsun)**(3./2.)

    radius = (numax/numaxref)*(delta_nu/deltanusun)**(-2) *(teff/teffsun)**(1./2.)

    # H_env = 2.3* 2.03e7*numax**(-2.38) # [ppm2/muHz], Mosser sf2a 2013 , mutlipied by 3 to have the true Hmax (??)
    H_env = 2.01e7 *numax**(-1.9) # [ppm2/muHz],  Peralta et al 2018
        
    
    # Mode line width, from Vrard et al 2018
    if( (mass >= 1.0) & (mass<1.2)):
        gamma = 0.1*(teff/4800.)**(4.55)
    elif( (mass >= 1.2) & (mass<1.4)) :
        gamma = 0.1*(teff/4800.)**(5.29)
    elif( (mass >= 1.4) & (mass<1.6)) :
        gamma = 0.1*(teff/4800.)**(4.54)
    elif( (mass >= 1.6) & (mass<1.8)) :
        gamma = 0.1*(teff/4800.)**(4.65)
    else:
        print ("UP:  scaling for mode line-width (Vrard et al 2018): stellar mass M=%g is out of range M=[1.2-1.8]" % (mass))
        print ("Belkacem et al's theoretical scaling relation is adopted")
        gamma = 0.19*(teff/4800.)**(10.8)   # Belkacem, sf2a, 2012

    
    width_env =  0.66*(numax**0.88)        # dnu_env = FWHM of the excess power envelope, see: Mosser et al. (2012 A&A 537 A30) - Table 2
    
    # building The Universal Pattern
    StarName =  ("%10.10i") % starID
    if(oscillation):
        if(rot_period_sur>0.):
            rot_env_f = 1./(rot_period_sur*86400.)*1e6
        else:
            rot_env_f = 0.
        if verbose:
            print(f'rot_env_f = {rot_env_f} muHz')
            print(f'rot_core_f = {rot_core_f} muHz')
        fout = open(path + StarName + '.modes','w')
        UPm,_ = UP.universal_pattern(delta_nu, numax, H_env,  f , nyq , verbose = verbose , teff = teff, DPI = DPI,
                                     q=q, gamma=gamma, width = width_env , rot_core_f = rot_core_f , beta = incl ,
                                     epsg = 0.25 , fout = fout,rot_env_f=rot_env_f)
        fout.close()
    else:
        UPm = 0.

    # Granulation spectrum (GS)
        # Granulation spectrum (GS)
    if(granulation):
        if ( (numax < 2.) | (numax > 3500.)):
            raise SLSError("Numax must be in the range of 2-3500 uHz if model is UP! ")
        GS, gran_par   = granulation_spectrum(f,numax,mass=mass,type=GST) 
        GSf,  gran_par  = granulation_spectrum(2.*nyq-f,numax,mass=mass,type=GST,verbose=verbose)  # Folded component of the GS
    else:
        GS,GSf = 0.,0.
        gran_par = {}

    if activity is None:
        AS = 0.
        ASf = 0.
        activity_par = {'sigma_a': 0., 'tau_a': 0.}
    else:
        AS = activity_spectrum(f,activity,verbose=verbose)
        ASf = activity_spectrum(2.*nyq-f,activity)
        activity_par = {'sigma_a': activity[0], 'tau_a': activity[1]}

    # mean power spectrum [ppm2/Hz]
    SC = 1e6 *(np.sinc(0.5*f/nyq))**2
    mpsd_osc = UPm * (b2m**2) * SC
    mpsd_gran = (GS  + GSf) * SC
    mpsd_act = (AS + ASf) * SC
    mpsd = 2.*W**2
    if(oscillation):
        mpsd += mpsd_osc
    if(granulation):
        mpsd += mpsd_gran
    if(activity is not None):
        mpsd += mpsd_act
    time, ts , ps , seed =  mpsd2rts(f*1e-6,mpsd,seed=seed)

    star_par = {'mass': mass, 'teff' :  teff, 'radius': radius}
    osc_par = {'numax' : numax, 'deltanu': delta_nu, 'Hmax' : H_env, 'Gammamax': gamma, 'width_envelop' : width_env,'inclination': incl, 'rot_core_f': rot_core_f }

    if(verbose):
        print (("numax [muHz], delta_nu [muHz], mass, Teff [K]: %f , %f, %f, %f") % (numax,delta_nu,mass,teff))
        print (("H_env [ppm2/muHz], width_env [muHz], gamma [muHz]: %e , %f , %f") % (H_env,width_env,gamma))
        print (("b2m, white_noise  [ppm2/muHz]: %f %g")% ( b2m,2.*W**2*1e-6 ))
        print (("times series, standard deviation [ppm]: %f") % np.std(ts))
        print (("theoretically expected standard deviation [ppm]: %f") %  math.sqrt(  np.sum(mpsd)*1e-6* f[0] )) 

    # output parameters
    opar = (star_par,obs_par,osc_par,gran_par,activity_par)

    return time,ts,f,ps*1e-6,mpsd*1e-6,mpsd_osc*1e-6,mpsd_gran*1e-6,mpsd_act*1e-6,opar,seed



def savelc(time,lc,StarID,par,seed,path='',navg=1,lctype=0):
    '''

    savelc(time,lc,StarID,par,seed,path='')
    
    Save the light-curve in an ASCII file compatible with kepler and the SSI pipeline (TypeFile=8)

    time : time values
    lc : the light-curve
    StarID : ID of star (StarID must be >0)
    par : simulation parameters 
    seed: seed associated with the simulation
    path: where to save the ligth-curve
    navg: input light-curve values are averaged over navg consecutive values (default: navg = 1, i.e. no average)
          the sampling of the output light-curves will then be equal to dt*navg
    lctype: type of ligt-curve format, 
        type=0 : KASOC ASCII format
        type=1 : simple ASCII format
    '''


    if (navg > 1 ):
        n = lc.size/navg
        t = rebin1d(time[0:n*navg],n)/float(navg)
        lc = rebin1d(lc[0:n*navg],n)/float(navg)
    else:
        t = time
        n = lc.size
    if(lctype==0):
        t = t/86400. # time in Julian days
    filename = path
    if(lctype==0):
        filename += ('kplr%10.10i') %  (StarID)
    else:
        filename += ('%10.10i') %  (StarID)

    f=open(filename+'.dat','w')
    f.write(('# seed  %i\n') % (seed))
    def ppar(par):
        n = len(par)
        i = 0 
        for u in list(par.items()):
            f.write(' %s = %g' % u)
            if (i < n-1):
                f.write(', ')
            i += 1
    f.write('\n')
    f.write('# star parameters:')
    ppar(par[0])
    f.write('# observations parameters:')
    ppar(par[1])
    f.write('# oscillation parameters:')
    ppar(par[2])
    f.write('# granulation parameters:')
    ppar(par[3])
    f.write('# activity parameters:')
    ppar(par[4])
    f.write('# navg: %i\n' % (navg))
    if(lctype==0):
        f.write('# Time [julian days], Flux [ppm], Flag\n')
        for i in range(n):
            f.write( ("%25.15e %25.15e   0\n") % (t[i],lc[i]) )
    else:
        f.write('# Time [s], Flux\n')
        for i in range(n):
            f.write( ("%25.15e %25.15e\n") % (t[i],lc[i]*1e-6+1.) )
        
        
    f.close()
  

def genslc(n,f,mpsd,par, seed=None,path='', verbose=False,navg = 1):
    '''    
    Generate several light-curves and save them in a set of files
    Each simulated light-curve corresponds to a given realization.
    
    seed  = genslc(n,numax,seed=None,path='', .... )

    n: number of simulated light-curves 
    f: frequency values [muHz] (as returned by gen_up or gen_adipls)
    mps: mean power spectrum [ppm2/muHz]  (as returned by gen_up or gen_adipls)
    par: simulation parameters (as returned by gen_up or gen_adipls)

    path: directory where the light-curves will be stored
    seed: initial seed of the Random Number Generator (RNG). If seed=None, choose one randolmy
    verbose: if true, verbose mode is on
        navg: light-curve values are  averaged over navg consecutive values (default: navg = 1, i.e. no average). 
              the sampling of the output light-curves will then be equal to dt*navg


    Example:
    seed = 13434343
    time,ts,f,ps,mps,par,seed = gen_up(StarID, numax,dt,T, mag , seed = seed )
    seed = genslc(n,f,mps,par,seed=seed)

    '''

    if( seed == None):
        # see is not specified, we choose a random value 
        seed = np.random.random_integers(0,high=2**30)    
    # initialization of the state of the RNG
    np.random.seed(seed)
    seeds = np.random.random_integers(0,high=2**30,size=n)
    for i in range (n):
        StarID = i+1
        time, ts , ps, seedi =  mpsd2rts(f*1e-6,mpsd*1e6,seed=seeds[i])

        savelc(time,ts,StarID,par,seedi,path=path,navg=navg)

    return seed


def ExtractSystematicDataMagRangeV2(filename,Vp,DriftLevel='medium',Verbose=False,seed=None):
    '''
    Vp: PLATO V reference magnitude (i.e.  V@T=6000K)
    '''
    if(Verbose):
            print('Loading the file %s that contains the parameters associated with the residual systematic error ' % filename)
    Data = np.load(filename,encoding="latin1",allow_pickle=True)
    Mag = Vp - 0.34 # conversion into P magnitude
    magcorr = 0.
    if(not (DriftLevel is None)):
        if(DriftLevel == 'low'):
            DriftAmpRange = [0.,0.4]
        elif(DriftLevel == 'medium'):
            DriftAmpRange = [0.4,0.8]
        elif(DriftLevel == 'high'):
            DriftAmpRange = [0.8,1e3]
        elif(DriftLevel == 'min'):
            DriftAmpRange = None
        elif(DriftLevel == 'max'):
            DriftAmpRange = None
        else: # any amplitude
            DriftAmpRange = [0.,1e3]
    n = 0
    DataOut = []
    try:
        if(Verbose):
            print('%i cameras simulated' % len(Data))    
        DataC1 = Data[0]
        p = len(DataC1) 
        IDC1 = np.zeros(p,dtype=np.int64)
        MagC1 = np.zeros(p)
        MaxDispC1 = np.zeros(p)
        i = 0
        for d in DataC1:
            IDC1[i] =  int(d['StarID'])  
            MagC1[i] =  d['Mag']  
            MaxDispC1[i] =  d['MaxDisp']  
            i+= 1
        if(Verbose):
            print('%i stars included' % i)
        if(DriftLevel == 'min' or DriftLevel == 'max'):
            m =  (MagC1+magcorr-0.25<=Mag) & (Mag<MagC1+magcorr+0.25)
            if(DriftLevel == 'min'): 
                v = np.min(MaxDispC1[m])
            else:
                v = np.max(MaxDispC1[m])
            m = m & (np.abs(MaxDispC1-v) <1e-5)
        else:
            m =  ((MagC1+magcorr-0.25<=Mag) & (Mag<MagC1+magcorr+0.25)  & (MaxDispC1>=DriftAmpRange[0]) & (MaxDispC1<DriftAmpRange[1]))
        nm = m.sum()
        if(Verbose):
            print('%i stars are matching the criteria' % nm)
        if(nm==0):
            raise SLSError('None of the stars in %s are matching the criteria' % filename)   
        np.random.seed(seed)
        j = np.random.randint(0, nm)
        StarIDData = IDC1[m][j]
        if(Verbose):
            print('Among them, StarID %9.9i has been randomly selected (Magnitude P= %f, Vp= %f) ' % (StarIDData,MagC1[m][j],MagC1[m][j]+0.34))
        ic = 0
        for C in Data:
            for s in C:
                if (s['StarID'] == StarIDData):
                    DataOut.append(s) 
                    if(Verbose):
                        SPRtot = s['SPRtot']
                        MaxDisp = s['MaxDisp']
                        MaskData = s['MaskData']
                        if(MaskData is not None):
                            MaskUpdNb = MaskData['MaskUpdNb']
                            print('Camera #%i: SPRtot = %f Max drift = %f, Mask number: %i' % (ic+1,SPRtot,MaxDisp,MaskUpdNb))
                        else:
                            print('Camera #%i: SPRtot = %f Max drift = %f' % (ic+1,SPRtot,MaxDisp))
                    n +=1
            ic += 1
        if(Verbose):
            print('%i individual light-curves for StarID %9.9i were found in %s ' %(n,StarIDData,filename))
    except SLSError:
        raise SLSError('Some errors occurs with %s' % filename)
    except:
    ## except KeyError:
        raise SLSError('Some errors occur with %s , are you sure about the data version (Systematics/Version parameter) ?' % filename)

    return DataOut


def ExtractSystematicDataMagRange(filename,Vp,version=2,DriftLevel='medium',Verbose=False,seed=None):
    '''
    Vp: PLATO V reference magnitude (i.e.  V@T=6000K)
    '''
    
    if(version ==0):
        magcorr = 0.218 # magnitude correction : original reference flux 2.64e5 e-/exp, corrected : 2.16e5, ratio : 1.222
        Mag = Vp
    elif(version ==1):
        magcorr = 0.
        Mag = Vp - 0.34 # conversion into P magnitude
        if(not (DriftLevel is None)):
            if(DriftLevel == 'low'):
                DriftAmpRange = [0.,0.4]
            elif(DriftLevel == 'medium'):
                DriftAmpRange = [0.4,0.8]
            elif(DriftLevel == 'high'):
                DriftAmpRange = [0.8,1e3]
            else: # any amplitude
                DriftAmpRange = [0.,1e3]
    elif(version==2):
        return ExtractSystematicDataMagRangeV2(filename,Vp,DriftLevel=DriftLevel,Verbose=Verbose,seed=seed)

    else:
        raise SLSError('Unknown version of file for the systematic error parameters')
    

    n = 0
    DataOut = []
    try:
        Data = np.load(filename,encoding="latin1",allow_pickle=True)
        if(version <1):
            for d in Data:
                if( (d['Mag']+magcorr-0.25<=Mag) and (Mag<d['Mag']+magcorr+0.25)):
                    DataOut.append(d)
                    n += 1
        else:
            for d in Data:
                if( (d['Mag']+magcorr-0.25<=Mag) and (Mag<d['Mag']+magcorr+0.25) 
                    and (d['MaxDisp']>=DriftAmpRange[0]) and (d['MaxDisp']<DriftAmpRange[1])):
                    DataOut.append(d)
                    n += 1
    except:
    ## except KeyError:
        raise SLSError('Some errors occur with %s , are you sure about the data version (Systematics/Version parameter) ?' % filename)
    if(n==0):
        raise SLSError('No systematic error parameters for magnitude Vp = %f \
        (PLATO V reference magnitude). Please turn-off systematic error or \
         change the star magnitude, the drift amplitude or use another table for the systematic errors' % Vp)
    elif(Verbose):
        print ('Systematic error parameters: number of sets of parameters matching the requirements: %i ' %n)
        
    if(n<24):
        print("WARNING ! The number of sets of systematic error parameters matching the requirements is low (<24). It is recommended to consider another drift level.")
        
    return DataOut


def GenLC(Data,t,seed=None,time_shift=0.,verbose=False,time_contraction=1.,quarter_duration=90.):
    '''

    lc,lcrn,rawlc,rawlcrn,t,Tupd,flag,LCmean,RawLCmean = GenLC(StarID,datapath,seed=None,time_shift=0.)
     
    Generate corrected and raw light-curves (LC) associated with a given star ID and given data set
     
    Inputs:
    
    StarID : the ID of the star for wich we want to generate the ligth-curve
    datapath : path to the data set
    time_shift: imposed time shift in seconds (default: 0.). Note that the CCD readouts are shifted by 6.25s
    For instance for CCD1, the time shift is 0, it is 6.25s for CCD2, 2*6.25s for CCD3 and 3*6.25 for CCD4.
    This option permits to introduce the time shift. 
    seed: assumed value of the random generator seed 
    (default: none, in that case the seed is generated on the basis of the current time)
    
    Outputs:
    lc[N]: corrected LC, N is the number of exposures
    lcrn[N]: corrected LC with random noise (RN)
    rawlc[N]: raw LC
    rawlcrn[N]: raw LC with RN 
    t[N]: exposure times
    Tupd[p]: times of the mask updates (p is the number of mask updates, the first mask used is considered as an update)
    flag[N]: flag the exposure at which a mask is updated  (or first time used)
    LCmean: mean value of the corrected LC
    RawLCmean: mean value of the raw LC
    
    '''
    

    # tau0 = 90. # simulated LC duration, in days
    # tau00 = tau0*86400.
    quarter_duration_s = quarter_duration*86400.
    tau0 = max(quarter_duration_s,90.*86400.)
    def simulate_segment(time,p):
        x = (time-time[0])/tau0
        return np.polyval(p,x) + 1.

    N = t.size
    np.random.seed(seed)
    
    lc = np.zeros(N)
    lcrn = np.zeros(N) 

    rawlc = np.ones(N)
    rawlcvar = np.ones(N)
    rawlcrn = np.ones(N)
        
    flag = np.zeros(N,dtype=np.bool_)
    
    Tupd = np.array([])


    MaskData = Data['MaskData']
    CorrectedLCData = Data['CorrectedLCData']
    if(verbose):
        SPRtot = Data['SPRtot']
        MaxDisp = Data['MaxDisp']
        print('SPRtot = %f' % SPRtot)
        print('Max drift = %f' % MaxDisp)
    RawLCData = Data['RawLCData']
    RawLCParameters =  RawLCData['parameter']
    SkyBackground = Data['SkyBackground']
    ReadOutNoise = Data['ReadOutNoise']
    Gain = Data['Gain']
    RawLCmean = RawLCData['LCmean']
    if(not (MaskData is None)):
        MaskSize = MaskData['MaskSize']

    if(MaskData is not None):
        MaskUpdNb = MaskData['MaskUpdNb']
        if(verbose):
            print('Number of mask udpates: %i' % MaskUpdNb)
        MaskUpdTime = MaskData['MaskUpTime']*time_contraction
##        print(MaskUpdNb,MaskUpdTime)
    else: # for PSF fitting method : no mask
        MaskUpdNb = 1
        MaskUpdTime = [0.]
        MaskSize = [0]
    Parameters =  CorrectedLCData['parameter']
    LCmean = CorrectedLCData['LCmean']
    # loop over the updates
    ncoef = int(Parameters.size/MaskUpdNb)
    t0 = 0.
    for i in range(MaskUpdNb):
        t1 = t0 + MaskUpdTime[i]
        if(i<MaskUpdNb-1):
            t2 = t0 + MaskUpdTime[i+1]
        else:
            t2 = t[-1] + 25.
                
        m = (t>=t1) & (t<t2)
        n =  m.sum()  
        if(n>0):   
            p = RawLCParameters[i*ncoef:(i+1)*ncoef]
            rawlc[m] = simulate_segment(t[m],p)*RawLCmean
            rawlcvar[m] = rawlc[m] + MaskSize[i]*(SkyBackground+ReadOutNoise**2+Gain/12.)
##            print(rawlc[m],MaskSize[i]*SkyBackground,MaskSize[i]*ReadOutNoise**2,MaskSize[i]*Gain/12.)
            rawlcrn[m] = rawlc[m] + np.random.normal(size=n)*np.sqrt(rawlcvar[m])
            p = Parameters[i*ncoef:(i+1)*ncoef]
            lc[m] = simulate_segment(t[m],p)*LCmean
            lcrn[m] = rawlcrn[m]*(lc[m]/rawlc[m])
            j = np.where(m)
            flag[j[0][0]] = True
            Tupd = np.append(Tupd,t1)
    if(time_shift>0.):
        lc = np.interp(t+time_shift,t,lc)

    return lc,lcrn,rawlc,rawlcrn,rawlcvar,t,Tupd,flag,LCmean,RawLCmean 


def SimSystematicErrorV2(Sampling,N,DataAllCamera,camera,quarter_duration,time_shift=0.,Verbose=False,
                         seed=None,AgeSlope=0.,LCSwitchingError=0.,Normalize=True,
                         AddRandomNoise=False,TimeContractionFactor=10.):
    
    '''
    Sampling: sampling time (in seconds), must be an integer number of 25s
    N: number of exposures
    DataAllCamera: Systematic Error data associated with a given target
    camera: camera number for which the residual LC is generated
    AgeSlope: long-term slope due to the instrument aging (ppm/days)
    LCSwitchingError: LC switching error (in %)        
    '''
    Nc = len(DataAllCamera)
    if( math.fabs(Sampling/25. % 1)>1e-5  ):
        print('The sampling must be an integer number of 25s')
        raise SLSError('The sampling must be an integer number of 25s')

    nc = int(Sampling/25.)
    if(nc> 1 and Verbose):
        print('Integrating %i individual exposure(s)' % nc)
    t = np.arange(N,dtype=np.float64)*Sampling # time in seconds
    
    
    lc = np.zeros(N) # normalized corrected LC
    rawlcvar = np.ones(N)  # variance of the normalized raw LC
        
    flag = np.zeros(N,dtype=np.bool_)
    
    # number of quarters
    duration = N*Sampling/86400. # duration in days
    Nquarter = len(quarter_duration) # int(math.ceil(duration/90.))
    if(Verbose):
        print('Number of quarter: %i' % Nquarter)
    Tupd = np.array([])
    np.random.seed(seed)
    LCSwitchingErrors = np.random.normal(size=Nquarter)*(LCSwitchingError/100.)
    LCSwitchingErrors[0] = 0.
    quarter_duration_sec = quarter_duration*86400.
    seeds =  np.random.randint(0, 1073741824 + 1,size=Nquarter)
    # p = int(math.floor(90.*86400.)/Sampling)
    # loop over the quarters
    t0 = 0. # start time of the current quarter
    for q in range(Nquarter):
        j = (q*6+camera) % Nc
        m = np.where( (t>=t0) & (t<t0+quarter_duration_sec[q]) )[0]
        t_q = t[m]-t0
        if(Verbose):
            print('Quarter: %i, Camera: %i' % (q,j))
        lc_q,lcrn_q,rawlc,rawlcrn,rawlcvar_q,t_q,Tupd_q,flag_q,LCmean,RawLCmean = GenLC(DataAllCamera[j],t_q,seed=seeds[q],
                                                                                        time_shift=time_shift,verbose=Verbose,
                                                                                        time_contraction=TimeContractionFactor,
                                                                                        quarter_duration=quarter_duration[q])
        if(AddRandomNoise):
            lc_q = lcrn_q
        # if( q%2 == 1):
        #     lc_q = lc_q[::-1] # mirroring the residual LC for odd quarter
        # cumulating nc exposures
        if(Normalize):
            lc_q = lc_q/LCmean 
            rawlcvar_q = rawlcvar_q/RawLCmean**2
        rawlcvar_q /= nc
        # t_q = rebin1d(t_q,n)/float(nc)
        # m = min(p,N-q*p)
        # t[q*p:q*p+m] = t_q[0:m] + q*p*Sampling
        # s = (1. - (AgeSlope*1e-6)*(t[q*p:q*p+m]/86400.))
        # lc[q*p:q*p+m] = (lc_q[0:m] + LCSwitchingErrors[q])*s
        # rawlcvar[q*p:q*p+m] = rawlcvar_q[0:m]*s**2
        # flag[q*p:q*p+m] = flag_q[0:m]
        s = (1. - (AgeSlope*1e-6)*(t[m]/86400.))
        lc[m] = (lc_q + LCSwitchingErrors[q])*s
        rawlcvar[m] = rawlcvar_q*s**2
        flag[m] = flag_q
        Tupd = np.append(Tupd,Tupd_q)
        t0 += quarter_duration_sec[q] # start time of the next quarter
    return lc,rawlcvar,t,Tupd,flag
        

def SimSystematicError(Sampling,N,Data,camera,quarter_duration,seed=None,time_shift=0.,version=2,Verbose=False):

    if(version==2):
        return SimSystematicErrorV2(Sampling,N,Data,camera,quarter_duration,time_shift=time_shift,
                         seed=seed,TimeContractionFactor=10.,Verbose=Verbose)
    def simulate_segment(time,p):
        tau0 = 90.*86400.
        x = (time-time[0])/tau0
        return np.polyval(p,x) + 1.
    
    # simulation duration, in days    
    if(version==0):
        simduration = 90. 
    else:
        simduration = 88.1 

    t = np.arange(N,dtype=np.float64)*Sampling # time in seconds
    
    np.random.seed(seed)

    lc = np.zeros(N) # normalized corrected LC
    if(version>0):
##        rawlc = np.zeros(N) # raw LC (normalized w.r.t mean flux)
        rawlcvar = np.ones(N) # raw LC variance (normalized w.r.t mean flux)
    else:
##        rawlc = None
        rawlcvar = np.ones(N)
        
    flag = np.zeros(N,dtype=np.bool_)
    
    # number of quarters
    duration = N*Sampling/86400. # duration in days
    Nquarter = int(math.ceil(duration/simduration))
    
    NData = len(Data)
    Tupd = np.array([])

    # loop over the quarters
    for q in range(Nquarter):
        j = np.random.randint(0,NData)
        try:
            if(version>0):
                MaskData = Data[j]['MaskData']
                CorrectedLCData = Data[j]['CorrectedLCData']
                RawLCData = Data[j]['RawLCData']
                RawLCParameters =  RawLCData['parameter']
                SkyBackground = Data[j]['SkyBackground']
                ReadOutNoise = Data[j]['ReadOutNoise']
                Gain = Data[j]['Gain']
                RawLCmean = RawLCData['LCmean']
                if(not (MaskData is None)):
                    MaskSize = MaskData['MaskSize']
            else:
                MaskData = Data[j]
                CorrectedLCData = Data[j]
            if(not (MaskData is None)):
                MaskUpdNb = MaskData['MaskUpdNb']
                MaskUpTime = MaskData['MaskUpTime']
            else: # for PSF fitting method : no mask
                MaskUpdNb = 1
                MaskUpTime = [0.]
                MaskSize = [36]
            Parameters =  CorrectedLCData['parameter']
        except KeyError:
            raise SLSError('Some errors occur with the systematic error data set, are you sure about the data version (Systematics/Version parameter) ?')

        # loop over the updates
        ncoef = int(Parameters.size/MaskUpdNb)
        t0 = q*simduration*86400.
        for i in range(MaskUpdNb):
            t1 = t0 + MaskUpTime[i]
            if(i<MaskUpdNb-1):
                t2 = t0 + MaskUpTime[i+1]
            else:
                t2 = t0 + simduration*86400.
                    
            m = (t>=t1) & (t<t2)   
            if(m.sum()>0):   
                if(version>0):
                    p = RawLCParameters[i*ncoef:(i+1)*ncoef]
                    rawlc = simulate_segment(t[m],p)
                    rawlcvar[m] = (rawlc + 
                                   MaskSize[i]*(SkyBackground+ReadOutNoise**2+Gain/12.)
                                   /RawLCmean)/RawLCmean*(25./Sampling)
                p = Parameters[i*ncoef:(i+1)*ncoef]
                ## p[ncoef-1] *= 3.
                ## print (p,t1,t2)
                lc[m] = simulate_segment(t[m],p)
                j = np.where(m)
                flag[j[0][0]] = True
                Tupd = np.append(Tupd,t1)
    if(time_shift>0.):
        lc = np.interp(t+time_shift,t,lc)
##        if(version>0):
##            rawlc = np.interp(t+time_shift,t,rawlc)

    return lc,rawlcvar,t,Tupd,flag


# def test_gen_up():
#     # ----------------------------------------------
#     # Simulation parameters
#     #
#     
#     # numax [muHz]
#     numax = 61.2
# 
#     # deltan_nu [muHz]
#     delta_nu  = 5.72
# 
#     # DPI [s], asymptotic period spacing
#     DPI = 232.20
# 
#     # coupling coefficient
#     q = 0.27
# 
#     # effective temperature
#     teff = 4800.
#     
#     # sampling time
#     dt = 512.
# 
#     # once generated, light-curve values are averaged over navg consecutive individual values (default: navg = 1, i.e. no average
#     # the sampling of the output light-curve will then be equal to dt*navg
#     navg = 1
# 
#     # duration in  days
#     T = 150. 
# 
#     # photon-noise reference level  [ppm.hour^(1/2)]
#     pn_ref = 7.7
# 
#     # non-photonic white-noise reference level [ppm.hour^(1/2)]
#     wn_ref = 0.
# 
#     # reference magnitude for the white-noise level 
#     mag_ref = 6.
# 
#     # star magnitude
#     mag = 12.
# 
#     # Seed of the RNG
#     seed = None
#     
#     StarID = 1
# 
#     #
#     # ----------------------------------------------
# 
#     time,ts,f,ps,mps,opar,seed = gen_up(StarID,numax,dt,T, mag , delta_nu = delta_nu, seed = seed, pn_ref = pn_ref, wn_ref = wn_ref , mag_ref = mag_ref , verbose = 1 , teff=teff, DPI = DPI , q = q , GST = 1)
# 
#         # save the light-curve
#     savelc(time,ts,StarID,opar,seed,navg=navg)
# 
#     plt.ioff()
# 
#     plt.figure(1)
#     plt.clf()
#     plt.plot(f,mps)
#     plt.xlabel(r'$\nu$ [$\mu$Hz]')
#     plt.ylabel(r'[ppm$^2$/$\mu$Hz]')
#     plt.axis(xmin=15,xmax=110)
#     plt.savefig('test_gen_up_1.png')
# 
#     plt.figure(2)
#     plt.clf()
#     plt.plot(f,ps)
#     plt.plot(f,mps,'r')
#     plt.xlabel(r'$\nu$ [$\mu$Hz]')
#     plt.ylabel(r'[ppm$^2$/$\mu$Hz]')
#     plt.loglog()
#     plt.axis(ymin=1,ymax=1e5,xmin=1e-1,xmax=300)
#     plt.savefig('test_gen_up_2.png')
# 
# 
#     plt.figure(3)
#     plt.clf()
#     plt.plot(time,ts)
#     plt.savefig('test_gen_up_3.pdf')
# 
# 
# #    plt.semilogy()
# #    plt.loglog()
# 
#     plt.show(block=False)
# 
#     s = eval(input('Type ENTER to finish'))
#     
#     return time,ts,f,ps,mps,opar
# 
# def test_gen_adipls():
#     # ----------------------------------------------
#     # Simulation parameters
#     #
#     
#     # effective temperature
#     teff = 5953.74
#     
#     # sampling time
#     dt = 32.
# 
#     # duration in  days
#     T = 150. 
# 
#     # photon-noise reference level  [ppm.hour^(1/2)]
#     pn_ref = 7.7
# 
#     # non-photonic white-noise reference level [ppm.hour^(1/2)]
#     wn_ref = 0.
# 
#     # reference magnitude for the white-noise level 
#     mag_ref = 6.
# 
#     # star magnitude
#     mag = 6.
# 
#     #  parameters a,b used for modelling surface effects
#     a,b = -0.004369 , 3.275559
#     
#     # Seed of the RNG
#     seed = None
#     
#     StarID = 2
# 
#     # Name of the .gsm generated by ADIPLS
#     fname = 'test.gsm'
# 
#     #
#     # ----------------------------------------------
#     
#     time,ts,f,ps,mps,opar,seed = gen_adipls(StarID, fname,teff,dt,T,mag,verbose=True,seed=seed,mag_ref=mag_ref,pn_ref=pn_ref,wn_ref=wn_ref, a=a,b=b,plot=True, rot_period_sur =  -1. , GST = 1 ) 
# 
#     # save the light-curve
#     savelc(time,ts,StarID,opar,seed)
# 
#     plt.ioff()
# 
# 
#     plt.figure(1)
#     plt.clf()
#     plt.plot(f,mps)
#     plt.xlabel(r'$\nu$ [$\mu$Hz]')
#     plt.ylabel(r'[ppm$^2$/$\mu$Hz]')
#     plt.axis(xmin=500,xmax=3000)
#     plt.savefig('test_gen_adipls_1.png')
# 
#     plt.figure(2)
#     plt.clf()
#     plt.plot(f,ps)
#     plt.plot(f,mps,'r')
#     plt.xlabel(r'$\nu$ [$\mu$Hz]')
#     plt.ylabel(r'[ppm$^2$/$\mu$Hz]')
#     plt.loglog()
#     plt.axis(ymin=1e-2,ymax=1e2,xmin=10,xmax=5000)
#     plt.savefig('test_gen_adipls_2.png')
# 
#     plt.figure(3)
#     plt.clf()
#     plt.plot(time,ts)
#     plt.savefig('test_gen_adipls_3.pdf')
# 
# 
# #    plt.semilogy()
# #    plt.loglog()
# 
#     plt.show(block=False)
# 
#     s = eval(input('Type ENTER to finish'))
#     
#     return time,ts,f,ps,mps,opar



## def test_single_mode():
    

##     # A :  mode amplitude  (arbitrary units)
##     A = 10.


##     #  nu0 :  mode frequency  [muHz]
##     nu0 = 3000.

##     # gamma : mode line-width  [muHz]
##     gamma = 10.

##     # white-noise (rms)
##     W = 0.0

##     # sampling time
##     dt = 50.

##     # duration in  days
##     T = 50. 

##     seed = 1999


##     #
##     gamma0=gamma*1e-6
##     nu00=nu0*1e-6 # mode frequency in Hz


##     T0 = T*86400.
##     # size of array of the time series
##     n = long(T0/(dt*2.))

##     # frequency resolution
##     dnu = 1./(2*n*dt)

##     # frequency values
##     f = (np.arange(n-1)+1)*dnu

##     # mode height
##     H = 2.*A**2/(math.pi*gamma0)

##     nyq = 1./(2.*dt) # Nyquist frequency

##     # mode profile (lorenztian profile):
##     x = 2.*(f-nu00)/gamma0
##     xp = 2.*(2.*nyq-f-nu00)/gamma0

##     # mean power spectrum
##     # the second term corresponds to the folded component
##     mpsd =   H/(1.+x**2) + H/(1.+xp**2) + 2.*W**2*dt


##     time, ts , ps =  mpsd2rts(f,mpsd,seed=seed)

##     print ("power spectrum, standard deviation: %f") % math.sqrt(np.sum(ps)*dnu)

##     print ("times series, standard deviation: %f") % np.std(ts)

##     plt.figure(0)
##     plt.clf()
##     plt.plot(time,ts)

##     plt.figure(1)
##     plt.clf()
##     plt.plot(f*1e3,ps)
##     plt.plot(f*1e3,mpsd)
##     plt.semilogy()

##     plt.show()
