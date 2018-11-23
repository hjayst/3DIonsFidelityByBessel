import numpy as np
import scipy
from scipy.integrate import simps
import scipy.io
from config import *
import time
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
import numexpr as ne
from scipy.special import jv

pool = Pool(16)

steps = 1000
def bessel_fun(n, phi):
    expression = lambda t:np.cos(n*t-phi.reshape(1,-1)*np.sin(t))*1/np.pi
    epsilon = 0
    i = 0
    Inte_ = 0
    while epsilon > 10**(-5) or i < 2:
        i += 1
        steps = 10000*i
        timelist = np.linspace(0, np.pi, steps).reshape(-1,1)
        dt = np.pi/steps
        Inte = simps(expression(timelist),dx=dt,axis=0)
        epsilon = np.abs(np.sum(Inte-Inte_))
        Inte_ = Inte
        ##print steps
        ##print epsilon
    return Inte
def Bessel(n_order, phi):
    """
    Create a [n, num_ions] matrix for store n-order bessel function
    """
    if n_order == 0:
        #return bessel_fun(0,phi)
	return jv(0,phi)
    if n_order == 1:
	return jv(1,phi)
        #return np.array([bessel_fun(0,phi),bessel_fun(1,phi)])
    output = np.zeros([n_order, np.shape(phi)[0]])
    for i in range(n_order):
        #output[i,:] = bessel_fun(i,phi)
	output[i,:] = jv(i,phi)
    """
    
    output[0,:] = bessel_fun(0,phi)
    output[1,:] = bessel_fun(1,phi)
    for i in range(n_order):
        if i < 2:
            continue
        else:
            #print i
            output[i,:] = 2*(i-1)*output[i-1,:]/(phi)-output[i-2,:]
    """
    return output

def alpha_Fun(nseg, tau, mu, ith, phiM, omega_K, N, Bessel_n):
    """
    nseg is number of segments and should be an int
    tau is the total gate time with float type
    mu is the detuning with float type
    ith is the ith ion
    phiM is R_0i*\Deltak plus additional phase error
    phi0 is R_1i*\Deltak
    omega_k shoud have the dimension[3*N]
    N is number of ions
    This function should related to Bessel function
    """
    #print mu
    number_mode = np.shape(omega_K.flatten())[0]
    expression1 =  lambda t0,t1,n,mu,omega_K,phiM:ne.evaluate("1/2.0*(exp(1.0j*t1*omega_K)*\
    ((-(mu+n*RF)*cos(phiM+t1*(mu+n*RF))+1.0j*omega_K*sin(phiM+t1*(mu+n*RF)))/((-omega_K+mu+n*RF)*(omega_K+mu+n*RF))+\
    ((mu-n*RF)*cos(phiM+t1*(mu-n*RF))-1.0j*omega_K*sin(phiM+t1*(mu-n*RF)))/((omega_K+mu-n*RF)*(omega_K-mu+n*RF)))-\
    exp(1.0j*t0*omega_K)*\
    ((-(mu+n*RF)*cos(phiM+t0*(mu+n*RF))+1.0j*omega_K*sin(phiM+t0*(mu+n*RF)))/((-omega_K+mu+n*RF)*(omega_K+mu+n*RF))+\
    ((mu-n*RF)*cos(phiM+t0*(mu-n*RF))-1.0j*omega_K*sin(phiM+t0*(mu-n*RF)))/((omega_K+mu-n*RF)*(omega_K-mu+n*RF))))")
    
    phiM_ = phiM + np.pi/2.0
    
    
    
    tseg = np.linspace(0,tau,nseg)
    DeltaT = tau/nseg
    alpha = np.zeros([number_mode,nseg], dtype=np.complex128)
    for indT in range(nseg):
        epsilon = 0
        n = 0
        Deltaf = 0
        t0 = tseg[indT]
        t1 = tseg[indT]+DeltaT
        Integration = 0
        while epsilon > 10**-5 or n < 2:
            
            if n == 0:
                f0 = Bessel_n[0,ith]*expression1(t0,t1,0,mu,omega_K,phiM)+2*Bessel_n[1,ith]*expression1(t0,t1,1,mu,omega_K,phiM_)
                Integration += f0
            else:
                fn = 2*(-1)**n*expression1(t0,t1,2*n,mu,omega_K,phiM)*Bessel_n[2*n,ith]\
                +2*(-1)**n*expression1(t0,t1,2*n+1,mu,omega_K, phiM_)*Bessel_n[2*n+1,ith]
                epsilon = np.sum(np.abs(fn/f0))
                Integration += fn
            n += 1
        alpha[:,indT] = Integration
    return alpha


def sum_Bessel(I1, I1_, I1_1, ith, jth, phiM1, phiM2, Bessel_n):
    #print "1"
    fn = 0
    #while epsilon > 10**-5 or m < 2:
    for m in range(3):
        for n in range(3):
            #print(m,n)
            if n==0 and m==0:


                I2 = Bessel_n[0,ith]*Bessel_n[0,jth]*I1_1(0,0,phiM1,phiM2)+\
                2*Bessel_n[0,ith]*Bessel_n[1,jth]*I1(0,1,phiM1,phiM2+np.pi/2)+\
                2*Bessel_n[1,ith]*Bessel_n[0,jth]*I1(1,0,phiM1+np.pi/2,phiM2)+\
                4*Bessel_n[1,ith]*Bessel_n[1,jth]*I1_(1,1,phiM1+np.pi/2,phiM2+np.pi/2)
                f0 = I2
                fn += I2
                #print(f0)
                #print "I2 is {0}".format(I2[0])
            elif n==0 and m != 0:
                I2 = (-1)**m*2*Bessel_n[2*m,ith]*Bessel_n[0,jth]*I1(2*m,0,phiM1,phiM2)+\
                (-1)**m*4*Bessel_n[2*m,ith]*Bessel_n[1,jth]*I1(2*m,1,phiM1,phiM2+np.pi/2)+\
                (-1)**m*2*Bessel_n[2*m+1,ith]*Bessel_n[0,jth]*I1(2*m+1,0,phiM1+np.pi/2,phiM2)+\
                (-1)**m*4*Bessel_n[2*m+1,ith]*Bessel_n[1,jth]*I1(2*m+1,1,phiM1+np.pi/2,phiM2+np.pi/2)
                fn += I2
                
                #print "I2 is {0}".format(I2[0])
            elif n != 0 and m == 0:
                I2 = (-1)**n*2*Bessel_n[2*n,jth]*Bessel_n[0,ith]*I1(0,2*n,phiM1,phiM2)+\
                (-1)**n*4*Bessel_n[2*n,jth]*Bessel_n[1,ith]*I1(1,2*n,phiM1+np.pi/2,phiM2)+\
                (-1)**n*2*Bessel_n[2*n+1,jth]*Bessel_n[0,ith]*I1(0,2*n+1,phiM1,phiM2+np.pi/2)+\
                (-1)**n*4*Bessel_n[2*n+1,jth]*Bessel_n[1,ith]*I1(1,2*n+1,phiM1+np.pi/2,phiM2+np.pi/2)

                fn += I2
                #epsilon = np.sum(np.abs(I2/f0))
                #if epsilon < 10**-5:
                    #break
                #print "I2 is {0}".format(I2[0])
            else:
		if m == n:
		         I2 = (-1)**(m+n)*4*Bessel_n[2*m,ith]*Bessel_n[2*n,jth]*I1_(2*m,2*n,phiM1,phiM2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m,ith]*Bessel_n[2*n+1,jth]*I1(2*m,2*n+1,phiM1,phiM2+np.pi/2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m+1,ith]*Bessel_n[2*n,jth]*I1(2*m+1,2*n,phiM1+np.pi/2,phiM2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m+1,ith]*Bessel_n[2*n+1,jth]*I1_(2*m+1,2*n+1,phiM1+np.pi/2,phiM2+np.pi/2)
		else:
                	I2 = (-1)**(m+n)*4*Bessel_n[2*m,ith]*Bessel_n[2*n,jth]*I1(2*m,2*n,phiM1,phiM2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m,ith]*Bessel_n[2*n+1,jth]*I1(2*m,2*n+1,phiM1,phiM2+np.pi/2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m+1,ith]*Bessel_n[2*n,jth]*I1(2*m+1,2*n,phiM1+np.pi/2,phiM2)+\
                	(-1)**(m+n)*4*Bessel_n[2*m+1,ith]*Bessel_n[2*n+1,jth]*I1(2*m+1,2*n+1,phiM1+np.pi/2,phiM2+np.pi/2)
                fn += I2
                #epsilon = np.sum(np.abs(I2/f0))
                #if epsilon < 10**-5:
                    #break  
                #print "I2 is {0}".format(I2[0])
    #print("error for one element is {0}".format(np.max(np.abs(I2/fn))))
    #print("ta,tb,tc,td is.{0},{1},{2},{3}, and p,q is {4},{5},the result is {6} ".format(ta,tb,tc,td,p,q,f0))
    return fn

def phifun(nseg, tau, mu, phiM1, phiM2, ith, jth, omega_K, Bessel_n):
    DeltaT = tau/nseg

    if type(omega_K) == int:
	omega_K = np.array([omega_K])
    tseg = np.arange(0,tau,DeltaT)
    #For [[ta,tb],[tc,td]] integrat
    expression = lambda ta,tb,tc,td,mu1,mu2,phi1,phi2,omega_K:ne.evaluate("1.0/((omega_K**2-mu1**2)*(omega_K**2-mu2**2))*\
    (exp(1.0j*(ta-tc-td)*omega_K)*(-cos(ta*mu1+phi1)*mu1+1.0j*sin(ta*mu1+phi1)*omega_K)\
     +exp(1.0j*(tb-tc-td)*omega_K)*(cos(tb*mu1+phi1)*mu1-1.0j*sin(tb*mu1+phi1)*omega_K))*\
    (-exp(1.0j*td*omega_K)*(cos(tc*mu2+phi2)*mu2+1.0j*sin(tc*mu2+phi2)*omega_K)+\
     exp(1.0j*tc*omega_K)*(cos(td*mu2+phi2)*mu2+1.0j*sin(td*mu2+phi2)*omega_K))")
    
    #For integation [[ta,tb],[tc,t]]
    """
    expression1 = lambda ta,tb,mu1,mu2,phi1,phi2,omega_K:ne.evaluate("(2*mu1*(mu1**2+omega_K**2+1.0j*mu1**2*omega_K*(ta-tb)-1.0j*omega_K**3*(ta-tb))*cos(phi1-phi2)+\
    mu1*(mu1**2-omega_K**2)*cos(phi1+phi2+2*mu1*ta)-2*exp(1.0j*omega_K*(-ta+tb))*\
    mu1**3*cos(phi1-phi2-mu1*ta+mu1*tb)-2*exp(1.0j*omega_K*(-ta+tb))*\
    mu1*omega_K**2*cos(phi1-phi2-mu1*ta+mu1*tb)+mu1**3*cos(phi1+phi2+2*mu1*tb)-\
    mu1*omega_K**2*cos(phi1+phi2+2*mu1*tb)-2*exp(1.0j*omega_K*(-ta+tb))*mu1**3*cos(phi1+phi2+mu1*(ta+tb))+\
    2*exp(1.0j*omega_K*(-ta+tb))*mu1*omega_K**2*cos(phi1+phi2+mu1*(ta+tb))-\
    4*1.0j*mu1**2*omega_K*sin(phi1-phi2)+2*mu1**4*ta*sin(phi1-phi2)-\
    2*mu1**2*omega_K**2*ta*sin(phi1-phi2)-\
    2*mu1**4*tb*sin(phi1-phi2)+\
    2*mu1**2*omega_K**2*tb*sin(phi1-phi2)-\
    1.0j*mu1**2*omega_K*sin(phi1+phi2+2*mu1*ta)+\
    1.0j*omega_K**3*sin(phi1+phi2+2*mu1*ta)+\
    4*1.0j*exp(1.0j*omega_K*(-ta+tb))*mu1**2*omega_K*sin(phi1-phi2-mu1*ta+mu1*tb)+\
    1.0j*mu1**2*omega_K*sin(phi1+phi2+2*mu1*tb)-\
    1.0j*omega_K**3*sin(phi1+phi2+2*mu1*tb))/(4*mu1*(mu1-omega_K)**2*(mu1+omega_K)**2)")
    """
    
    expression1 =  lambda ta,tb,mu1,mu2,phi1,phi2,omega_K:ne.evaluate("(-(mu1-omega_K)*(mu1+omega_K)*cos(mu2*ta)*(cos(phi1)*(sin(phi2)*(1.0j*mu1*omega_K*cos(mu1*ta)+mu2**2*sin(mu1*ta))+\
    mu2*cos(phi2)*(mu1*cos(mu1*ta)-1.0j*omega_K*sin(mu1*ta)))+\
    sin(phi1)*(cos(phi2)*(-1.0j*mu2*omega_K*cos(mu1*ta)-mu1*mu2*sin(mu1*ta))+\
    sin(phi2)*(mu2**2*cos(mu1*ta)-\
    1.0j*mu1*omega_K*sin(mu1*ta))))-(mu1-omega_K)*(mu1+\
    omega_K)*(cos(phi1)*(cos(phi2)*(1.0j*mu1*omega_K*cos(mu1*ta)+mu2**2*sin(mu1*ta))+\
    mu2*sin(phi2)*(-mu1*cos(mu1*ta)+1.0j*omega_K*sin(mu1*ta)))+\
    sin(phi1)*(mu2*sin(phi2)*(1.0j*omega_K*cos(mu1*ta)+mu1*sin(mu1*ta))+\
    cos(phi2)*(mu2**2*cos(mu1*ta)-\
    1.0j*mu1*omega_K*sin(mu1*ta))))*sin(mu2*ta)+(mu1-mu2)*(mu1+mu2)*(mu1*cos(phi1+mu1*ta)-\
    1.0j*omega_K*sin(phi1+mu1*ta))*(mu2*cos(phi2+mu2*ta)+\
    1.0j*omega_K*sin(phi2+mu2*ta))+(mu1-omega_K)*(mu1+\
    omega_K)*cos(mu2*tb)*(cos(phi1)*(sin(phi2)*(1.0j*mu1*omega_K*cos(mu1*tb)+mu2**2*sin(mu1*tb))+\
    mu2*cos(phi2)*(mu1*cos(mu1*tb)-1.0j*omega_K*sin(mu1*tb)))+\
    sin(phi1)*(cos(phi2)*(-1.0j*mu2*omega_K*cos(mu1*tb)-mu1*mu2*sin(mu1*tb))+sin(phi2)*(mu2**2*cos(mu1*tb)-\
    1.0j*mu1*omega_K*sin(mu1*tb))))+(mu1-omega_K)*(mu1+omega_K)*(cos(phi1)*(cos(phi2)*(1.0j*mu1*omega_K*cos(mu1*tb)+mu2**2*sin(mu1*tb))+\
    mu2*sin(phi2)*(-mu1*cos(mu1*tb)+1.0j*omega_K*sin(mu1*tb)))+sin(phi1)*(mu2*sin(phi2)*(1.0j*omega_K*cos(mu1*tb)+mu1*sin(mu1*tb))+\
    cos(phi2)*(mu2**2*cos(mu1*tb)-\
    1.0j*mu1*omega_K*sin(mu1*tb))))*sin(mu2*tb)+exp(1.0j*omega_K*(-ta+tb))*(mu1-mu2)*(mu1+\
    mu2)*(1.0j*mu2*cos(phi2+mu2*ta)-omega_K*sin(phi2+mu2*ta))*(1.0j*mu1*cos(phi1+mu1*tb)+\
    omega_K*sin(phi1+mu1*tb)))/((mu1-mu2)*(mu1+mu2)*(mu1-omega_K)*(mu2-omega_K)*(mu1+omega_K)*(mu2+omega_K))")


    expression2 = lambda ta,tb,mu1,phi1,phi2,omega_K:ne.evaluate("(2*mu1*(mu1**2+omega_K**2+1.0j*mu1**2*omega_K*(ta-tb)-1.0j*omega_K**3*(ta-tb))*cos(phi1-phi2)+\
    mu1*(mu1**2-omega_K**2)*cos(phi1+phi2+2*mu1*ta)-2*exp(1.0j*omega_K*(-ta+tb))*\
    mu1**3*cos(phi1-phi2-mu1*ta+mu1*tb)-2*exp(1.0j*omega_K*(-ta+tb))*mu1*omega_K**2*cos(phi1-phi2-mu1*ta+mu1*tb)+\
    mu1**3*cos(phi1+phi2+2*mu1*tb)-mu1*omega_K**2*cos(phi1+phi2+2*mu1*tb)-\
    2*exp(1.0j*omega_K*(-ta+tb))*mu1**3*cos(phi1+phi2+mu1*(ta+tb))+\
    2*exp(1.0j*omega_K*(-ta+tb))*mu1*omega_K**2*cos(phi1+phi2+mu1*(ta+tb))-\
    4*1.0j*mu1**2*omega_K*sin(phi1-phi2)+2*mu1**4*ta*sin(phi1-phi2)-\
    2*mu1**2*omega_K**2*ta*sin(phi1-phi2)-\
    2*mu1**4*tb*sin(phi1-phi2)+\
    2*mu1**2*omega_K**2*tb*sin(phi1-phi2)-\
    1.0j*mu1**2*omega_K*sin(phi1+phi2+2*mu1*ta)+\
    1.0j*omega_K**3*sin(phi1+phi2+2*mu1*ta)+\
    4*1.0j*exp(1.0j*omega_K*(-ta+tb))*mu1**2*omega_K*sin(phi1-phi2-mu1*ta+mu1*tb)+\
    1.0j*mu1**2*omega_K*sin(phi1+phi2+2*mu1*tb)-\
    1.0j*omega_K**3*sin(phi1+phi2+2*mu1*tb))/(4*mu1*(mu1-\
    omega_K)**2*(mu1+omega_K)**2)")


    phi_kij = np.zeros([np.shape(omega_K.flatten())[0],nseg,nseg], dtype=np.complex256)  
    for p in range(nseg):
        ##print p
        for q in range(p):
            ta = tseg[p]
            tb = tseg[p]+DeltaT
            tc = tseg[q]
            td = tseg[q]+DeltaT
            #print("p is {0}, q is {1}".format(p,q))
            #print("ta,tb,tc,td is.{0},{1},{2},{3}".format(ta,tb,tc,td))
            I1 = lambda m_,n_,phi1,phi2:1/4.0*(expression(ta,tb,tc,td,mu+m_*RF,mu+n_*RF,phi1,phi2,omega_K)+\
            expression(ta,tb,tc,td,mu+m_*RF,mu-n_*RF,phi1,phi2,omega_K)+\
            expression(ta,tb,tc,td,mu-m_*RF,mu+n_*RF,phi1,phi2,omega_K)+\
            expression(ta,tb,tc,td,mu-m_*RF,mu-n_*RF,phi1,phi2,omega_K))

            phi_kij[:,p,q] = sum_Bessel(I1,I1,I1,ith,jth,phiM1,phiM2,Bessel_n)+sum_Bessel(I1,I1,I1,jth,ith,phiM2,phiM1,Bessel_n)
        ta = tseg[p]
        tb = tseg[p]+DeltaT
        #print("p is {0}, q is {1}".format(p,p))
	#print("ta,tb,tc,td is.{0},{1},{2}".format(ta,tb,ta))
	
	I1 = lambda m_,n_,phi1,phi2:1/4.0*(expression1(ta,tb,mu+m_*RF,mu+n_*RF,phi1,phi2,omega_K)+\
        expression1(ta,tb,mu+m_*RF,mu-n_*RF,phi1,phi2,omega_K)+\
        expression1(ta,tb,mu-m_*RF,mu+n_*RF,phi1,phi2,omega_K)+\
        expression1(ta,tb,mu-m_*RF,mu-n_*RF,phi1,phi2,omega_K))
	I1_ = lambda m_,n_,phi1,phi2:1/4.0*(expression2(ta,tb,mu+m_*RF,phi1,phi2,omega_K)+\
        expression1(ta,tb,mu+m_*RF,mu-n_*RF,phi1,phi2,omega_K)+\
        expression1(ta,tb,mu-m_*RF,mu+n_*RF,phi1,phi2,omega_K)+\
        expression2(ta,tb,mu-m_*RF,phi1,phi2,omega_K))        

        I1_1 = lambda m_,n_,phi1,phi2:1/4.0*(expression2(ta,tb,mu+m_*RF,phi1,phi2,omega_K)+\
        expression2(ta,tb,mu+m_*RF,phi1,phi2,omega_K)+\
        expression2(ta,tb,mu-m_*RF,phi1,phi2,omega_K)+\
        expression2(ta,tb,mu-m_*RF,phi1,phi2,omega_K))
	

	phi_kij[:,p,p] = sum_Bessel(I1,I1_,I1_1,ith,jth,phiM1,phiM2,Bessel_n)+sum_Bessel(I1,I1_,I1_1,jth,ith,phiM2,phiM1,Bessel_n) 

	
    return phi_kij




def Avg_FID(Omega_i, tau, muDel, alpha_ki, alpha_kj, Phi, ith, jth, betak, gk, Flag):
    alpha_ki = np.dot(alpha_ki,Omega_i)*gk[ith].reshape(-1,1)
    alpha_kj = np.dot(alpha_kj,Omega_i)*gk[jth].reshape(-1,1)
    index1 = 2*np.real(np.dot(betak.T,alpha_ki*np.conjugate(alpha_ki)))
    index2 = 2*np.real(np.dot(betak.T,alpha_kj*np.conjugate(alpha_kj)))
    indexCross = 2*np.real(np.dot(betak.T,alpha_kj*np.conjugate(alpha_ki)))
    Gamma1 = np.exp(-index1)
    Gamma2 = np.exp(-index2)
    Gammap=np.exp(-index1-index2-2*(indexCross))
    Gammam=np.exp(-index1-index2+2*(indexCross))
    Phi_ij = np.dot(Omega_i.T,np.dot(np.sum((gk[ith,:]*gk[jth,:]).reshape(-1,1,1)*Phi,axis=0), Omega_i))
    DeltaE = 2*np.imag(np.sum(alpha_ki*np.conjugate(alpha_kj)))
    #print(np.shape(Gamma1))
    fidelity = (4+Flag*2*(Gamma1+Gamma2)*np.sin(2*Phi_ij+DeltaE)+Gammap+Gammam)/10.0
    return fidelity


if __name__ == "__main__":
    
    Bessel_n = np.load("Bessel_n.txt.npy")
    #Bessel_n[1] = Bessel(10, )
    print("phifun(nseg=14, tau=300*10**(-6), mu=10**7, phiM1=1, phiM2=2, ith=1, jth=4, omega_K=10**6, Bessel_n=Bessel_n)")
    print phifun(nseg=14, tau=300*10**(-6), mu=10**7, phiM1=1, phiM2=2, ith=1, jth=4, omega_K=10**6, Bessel_n=Bessel_n)
    
