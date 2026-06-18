'''

       This file is part of the Stellar Seismic Indices pipeline

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Copyright (C) 2016 by R. Samadi and R. Peralta
'''
from __future__ import print_function

from builtins import range
import numpy as np
import math
import scipy.optimize
## import ssilib



def gaussenvelop(nu,A,nu0,width):
    # width: full width at half maximum (delta_env)
        return A*np.exp(-((nu-nu0)**2)/(width**2/(4*np.log(2)))) 
        

"""
INPUT
deltanu : large separation [muHz]
numax    : peak frequency [muHz]
Ampli    : Amplitude of the gaussian envelope [ppm^2/muHz]
f    : frequency vector [muHz]

OUTPUT
A    : synthetic spectra

"""

# Ref. M13 : Mosser et al 2013, SF2A
numaxref= 3104.

# sun cutt-off freq.
nucsun = 5300.

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


def mixed_modes(nnup,nup,dnu,DPI,q,epsg=0.,verbose=False):
    '''
    Computed the mixed mods frequencies, periods, and other characteristics
    
    (num,Pm,Pg,nm,ng,zeta) = mixed_modes(nnup,nup,dnu,DPI,q,epsg=0.,verbose=False)
     
     nnup : radial order of the pure l=1 mode
     nup : frequency of the pure l=1 mode [muHz]
     dnu : asymptotic large separation [muHz]
     DPI : asymptotic period spacing [s]
     q   : coupling coefficient
     epsg:
    
     return (num,Pm,Pg,nm,ng,zeta)
     num : mixe modes freq. [muHz]
     Pm : mixed mode periods [s]
     Pg : associated pur g-mode period [s]
     nm : mixed mode radial orders
     ng : g mode raidal orders
     zeta : (Icore/I) where I is the mode inertia
     
   ''' 


    def mixed_mode_period_find(nup,dnu,q,DPI,epsg,nu1,nu2,eps=1e-7):
        P = 1e6/nu2
        P1 = 1e6/nu1
        roots = []
        x = nup +  (dnu/math.pi)*math.atan(q*math.tan(math.pi*(P/DPI-epsg)) ) - 1e6/P
        while ( (P <= P1) ) :
            P0 = P * (1. + eps )
            y = nup +  (dnu/math.pi)*math.atan(q*math.tan(math.pi*(P0/DPI-epsg)) ) - 1e6/P0
            if ( x*y < 0):
                # we refine the solution
                Pr = P - x* eps*P/(y-x)
                yn = nup +  (dnu/math.pi)*math.atan(q*math.tan(math.pi*(Pr/DPI-epsg)) ) - 1e6/Pr
#                nu0 = 1./(1./nur - DPI/10.*1e-6)
                if( math.fabs(yn) < 1e-5*nup):
##                    print Pr,P, P0,x,y,yn
                    roots.append(Pr)
            P = P0
            x = y
        return roots[::-1]
            

##    num = mixed_mode_freq_find(nup,q,DPI,epsg,nup-dnu/2.,nup + dnu/2.)
##    Pm = 1e6/num # mixed-mode periods (s)

    Pm = np.array(mixed_mode_period_find(nup,dnu,q,DPI,epsg,nup-dnu/2.,nup + dnu/2.))
    Nm =len(Pm)

    if(Nm <=0):
        return  (None,None,None,None,None,None) 

    num = 1e6/Pm
    
    ngp =  -int(math.floor(1e6/DPI/(nup) - epsg)) # radial order of the g-mode close to the pure p-mode (nup)
    Pgp = (- ngp  + epsg)*DPI # period of this g-mode
    if(verbose):
        print ('nup,np,ngp,Pg: %f %i %i %f' % (nup,nnup,ngp,Pgp))
    
    i = np.argmin( np.abs(num-nup) )
    ng = ngp + (np.arange(Nm)-i) # g-mode orders
    Pg = (- ng + epsg )*DPI # pure g-mode periods [s]
    nm  = ng + nnup  # mixed-mode orders

    #  zeta = Icore/I,  Goupil et al 2013
    # alpha0 = dnu*1e-6*DPI # Eq. A24 Goupil et al 2013
    # chi = 2.*(num/dnu)*np.cos(math.pi/DPI/num*1e6)
    # zeta = 1./(1. + alpha0 *chi**2 ) #  Eq. A27-A28 Goupil et al 2013
    ## zeta = 1./(1. + (num**2*DPI*1e-6)/(q*dnu)*(np.cos(math.pi*(1e6/num-Pg)/DPI))**2/(np.cos(math.pi*(num-nup)/dnu))**2) # Eq. 14 Mosser et al 2015
    theta = math.pi*(num-nup)/dnu
    zeta = 1./(1  + (num**2*DPI*1e-6)/(q*dnu)/(1./q**2*(np.sin(theta))**2+(np.cos(theta))**2)) # Eq. 4 Gehan et al 2018
    '''
    figure(200)
    clf()
    plot(zeta)
    plot(zeta2,'k+')
    figure(201)
    clf()
    plot(zeta,zeta2)
    
    draw()
    show(block=True)
    '''  
    if (verbose):
        for i in range(Nm):
             print ('num,Pm,Pg,ng,nm,zeta: %f %e %e %i %i %f' %  (num[i],Pm[i],Pg[i],ng[i],nm[i],zeta[i]))

    return (num,Pm,Pg,nm,ng,zeta)
        

def universal_pattern_modes(df, delta_nu, numax, Ampli, norders = None, verbose = False , \
                             teff=4800. , DPI = -1. , q = 0. , eps = -10., d01 = 10., d02 = -10. , \
                             gamma = -1. ,  rot_core_f = 0. , beta = 0. , alpha = -1., h_threshold  = 1e-2 , \
                             width=-1. , epsg = 0., V1 = -1. , rot_env_f = 0.):
    '''
    
    
    h_threshold : threshold in (normalised) height applied to the mixed modes.  
    beta: inclination angle (in degree)
    
    '''
    # Values given by Benoit 05.07.18:
    if(eps<-9.):
        eps = 0.6 + 0.52*math.log10(delta_nu) + delta_nu*0.005
        
    if(d01 >9.):
        d01 = 0.0553 - 0.036*math.log10(delta_nu)            
        
    if(d02 <-9.):
            d02 = 0.167 - 0.039*math.log10(delta_nu)
            
    d03 = 0.282*delta_nu + 0.16 # from Huber et al 2010  

    # Values from Mosser et al (2013)
    if(alpha<0.):
        alpha0 = 0.015*(delta_nu)**(-0.32)
    else:
        alpha0 = alpha
    alphal = (alpha0,alpha0,alpha0,alpha0)
     
    d0l=(0.,d01,d02,d03)
    
    if(gamma <0.):
        # Mode linewidths
        if ( (teff >= 5300) & (teff <=6400) & (numax > 600.) ):
            # Appourchaux et al (2012), table 2, at maximum mode height
            gamma0 = 0.20 # [muHz]
            gamma1 = 0.97  # [muHz]
            gamma_slope = 13.0
            gamma = gamma0 + gamma1*(teff/5777.)**gamma_slope
    
        else:
            # Belkacem, sf2a, 2012
            gamma = 0.19*(teff/4800.)**(10.8) #  *((numax/100.)*sqrt(teff/4800.) )**(-0.3)
    
    # Mode square visibility (ref. Mosser et al. 2012 A&A, 537, A30) 
    # Visibility for pure p modes in a RGB with Teff=4800 K  (Tab. 4)
    if(V1 <0.):
        V1 = 1.54  # Value given by Benoit 05.07.18
    V2 = 0.64
    V3 = 0.071
    V = (1.,V1,V2,V3)
    
#    nmax = int(round(numax/delta_nu-eps))    # Order n for which we obtain the maximum of oscillations
    nmax = (numax/delta_nu-eps)    # non-integer Order n for which we obtain the maximum of oscillations
    nmaxi = int(round(nmax))    # corresponding integer order n

    nuc = nucsun*(numax/numaxref)         # cut-off frequency

    if( norders == None):
        n1 = 1
        norders = int(nuc/delta_nu)        # Total number of the order n of the serie
        n2 = norders + 1
    else:
        n1 = nmaxi - norders/2 
        n2 = nmaxi +  norders/2 


    Nbr_l = 4            # Total number of the degree l of the serie

    L = np.array([]) # l values
    M = np.array([]) # m values
    N = np.array([]) # radial order
    G = np.array([]) # line-widths
    H = np.array([]) # mode heights
    F = np.array([]) # mode frequencies

    # The mode profiles are modulated by a gaussian envelope. For Gaussian parameters, see Mosser et al. (2012 A&A 537 A30)
    if(width<=0.):
        width = 0.66*(numax**0.88)        # dnu_env = FWHM of the excess power envelope, see: Mosser et al. (2012 A&A 537 A30) - Table 2
    
  
    if (verbose):
        print (("numax= %f, nuc= %f, nnmax= %f, nnuc= %i") % (numax,nuc,nmax,norders))
    for l in range(0,Nbr_l): # Loop angular degree l
        for n in range(n1-(l==2),n2+1-(l==2)):  # Loop over radial orders n
            nu = (n + (l/2.) + eps - d0l[l] + (alphal[l]/2.) * (n-numax/delta_nu )**2) * delta_nu    #in [muHz]    #Mosser et al. 2011
            if ( (l ==1) & (DPI >0.)  ):
                # we include mixe-modes (dipolar modes only)
                (num,Pm,Pg,nm,ng,zeta) = mixed_modes(n,nu,delta_nu,DPI,q,epsg,verbose=verbose)

                if( num is None):
                    break
                i = 0 
                for nu in num: # Loop over the mixed-modes
                    gammam = gamma*(1-zeta[i])
                    h = gaussenvelop(nu,Ampli,numax,width)
                    if ( gammam > df*2.): # resolved mode
                        pass
                    else: # un-resolved mode
                        h *= (math.pi*gammam/2./df)
                    if( h>h_threshold*Ampli ):                    
                        if (verbose):
                            print (("nm, num, gammam, zeta,h: %i %f %f %f %f") % (nm[i],nu,gammam,zeta[i],h))
                        # d_split = 0.5*zeta[i]*math.fabs(rot_core_f) # splitting in muHz
                        d_split = 0.5* (zeta[i]*(rot_core_f - 2.*rot_env_f) + 2.*rot_env_f) # Eq. 21  & 22 in Goupil et al 2013
                        if(d_split>0.):
                            pr = power_ratio(l,math.fabs(beta)) # power ratios of the mutliplets
                            for m in range(-l, l+1) : 
                                F=np.append(F,nu + m*d_split)
                                L=np.append(L,l)
                                M=np.append(M,m)
                                G=np.append(G,gammam)
                                N=np.append(N,n)
                                H=np.append(H,V[l]*h*pr[m+l])
                        else:           # no rotation (or  negligeable)            
                            F=np.append(F,nu)
                            L=np.append(L,l)
                            M=np.append(M,0)
                            G=np.append(G,gammam)
                            N=np.append(N,n)
                            H=np.append(H,V[l]*h)
                    i += 1
            else:
                h = gaussenvelop(nu,Ampli,numax,width)        
                if ( gamma > df*2.): # resolved profile
                    pass
                else: # un-resolved profile
                    h *= (math.pi*gamma/2./df)
                    
                d_split = math.fabs(rot_env_f) # splitting in muHz
                if( (d_split > 0.) & (l>0) ):
                    pr = power_ratio(l,math.fabs(beta)) # power ratios of the mutliplets
                    for m in range(-l, l+1) : 
                        F=np.append(F,nu + m*d_split)
                        L=np.append(L,l)
                        M=np.append(M,m)
                        G=np.append(G,gamma)
                        N=np.append(N,n)
                        H=np.append(H,V[l]*h*pr[m+l])
                else:
                    F=np.append(F,nu)
                    L=np.append(L,l)
                    M=np.append(M,0)
                    G=np.append(G,gamma)
                    N=np.append(N,n)
                    H=np.append(H,V[l]*h)
        
    return (F,L,M,N,H,G) 


def universal_pattern_new(delta_nu, numax, Ampli, f, nyquist, norders = None, verbose = False, teff=4800., \
                      DPI = -1., q = 0.,  eps = -10., d01 = 10., d02=-10.  ,  gamma = -1. , rot_core_f = 0. , beta = 0. , alpha = -1., h_threshold = 1e-3 , width = -1. , epsg = 0.):
        

    if (len(f) > 1):
        df = f[1]-f[0]    # Frequency resolution
    else:
        df = 1e-5

    (F,L,M,N,H,G)  = universal_pattern_modes(df, delta_nu, numax, Ampli, norders = norders ,  verbose = verbose , teff=teff \
                                               , DPI = DPI , q = q , eps = eps , d01 = d01 , d02 = d02 , gamma = gamma , \
                                               alpha = alpha ,  rot_core_f = rot_core_f , beta = beta , h_threshold =  h_threshold , width = width , \
                                               epsg = epsg )

    '''
    the following part is deprecated by the cython version

    m = len(F)
    
    if (verbose):
        print "n   l   nu [muHz]"
            
    for k in range(m):  # Loop over the modes
                nu =  F[k]    #  freq.
                n = N[k] #  order
                l = L[k] # angular degree l
                gammak = G[k] # line-widths
                h = H[k] # height

                if (verbose):
                        print ("n,l,nu,h,g: %i %i %f %f %f") % (n,l,nu,h,gammak)

                if ( gammak > df*2.): # resolved profile => Lorentzian : modes shape
                    Pr = 1./ ( 1. + ( 2.*(f-nu)/gammak)**2 ) #+ 1./ ( 1. + ( 2.*( (2.*nyquist - f) - nu )/gamma)**2 )
                else: # un-resolved profile => sinc shape
                    Pr = ( np.sinc( math.pi*(f-nu)/df) )**2 #+ ( np.sinc( math.pi*( (2.*nyquist - f)-nu )/df) )**2

                A = A+ h*Pr
    '''

    return universal_pattern_mode_profiles(df,f,F,L,N,H,G,verbose=verbose)

    
    
    

def universal_pattern(delta_nu, numax, Ampli, f, nyquist, norders = None, verbose = False, teff=4800., \
                      DPI = -1., q = 0.,  eps = -10., d01 = 10., d02=-10.  ,  gamma = -1. , rot_core_f = 0. , \
                      beta = 0. , alpha = -1., h_threshold = 1e-2 , width = -1. , epsg = 0. , V1 = -1. , fout  = None, \
                      rot_env_f=0.):
        
    A = np.zeros(f.size)

    if (len(f) > 1):
        df = f[1]-f[0]    # Frequency resolution
    else:
        df = 1e-5

    (F,L,M,N,H,G)  = universal_pattern_modes(df, delta_nu, numax, Ampli, norders = norders ,  verbose = verbose , teff=teff \
                                               , DPI = DPI , q = q , eps = eps , d01 = d01 , d02 = d02 , gamma = gamma , \
                                               alpha = alpha ,  rot_core_f = rot_core_f , beta = beta , h_threshold =  h_threshold , width = width , \
                                               epsg = epsg, V1 = V1 , rot_env_f=rot_env_f)



    nm = len(F)
      
    if( fout is not None):
        fout.write("# n  l  m  nu H Gamma\n")
  
    if (verbose):
        print ("# n l m  nu H Gamma")
            
    for k in range(nm):  # Loop over the modes
                nu =  F[k]    #  freq.
                n = N[k] #  order
                m = M[k]
                l = L[k] # angular degree l
                gammak = G[k] # line-widths
                h = H[k] # height

                if (verbose):
                        print (("%i %i %i %f %f %f") % (n,l,m,nu,h,gammak))
                if( fout is not None):
                    fout.write(("%i %i %i %f %f %f\n") % (n,l,m,nu,h,gammak))


                if ( gammak > df*2.): # resolved profile => Lorentzian : modes shape
                    Pr = 1./ ( 1. + ( 2.*(f-nu)/gammak)**2 ) #+ 1./ ( 1. + ( 2.*( (2.*nyquist - f) - nu )/gamma)**2 )
                else: # un-resolved profile => sinc shape
                    # note that np.sinc() already contains the pi factor
                    Pr = ( np.sinc( (f-nu)/df) )**2 #+ ( np.sinc( ( (2.*nyquist - f)-nu )/df) )**2

                A = A+ h*Pr
    return A,(F,L,M,N,H,G)

    


