import numpy as np
import scipy
from scipy.integrate import simps
import scipy.io
from config import *
import time
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
from Bessel_micromotion import *
from functools import partial

indmax = 16 #number of threads



data = np.load("data.npy")
data = data[()]
kDel = 2*2*np.pi/wavelength
#(omega_K, b) = findmodeAll(iterR)
omega_K = data["mode"]
b = data["b"]
B_new = data["B_new"]
#N = np.shape(B_new)[0]
print("number of ions {0}".format(N))
#tau = 300*10**(-6)
kDel = np.array([0,0,kDel])#only consider the z
#RF = 40*10**6*2*np.pi
#print np.shape(omega_K)
#omega_K = omega_K*RF/2.0
b_z = np.sum(b.T.reshape(np.shape(omega_K)[0],N,3)*kDel.reshape(1,1,-1),axis=2)#[num_mode, num_ions]
Etak = np.sqrt(const_hbar/(2*const_m*omega_K))#[num_mode]
gk = b_z.T*Etak.reshape(1,-1)#[num_ions, num_modes]
#modespan = ((max(omega_K)-min(omega_K)))/2000

#modespan = 0
#print("mode span is {0} MHz".format(modespan/10**6))


#muDelAll = np.arange(2.5*10**7,3.5*10**7,2*np.pi*10**3)
muDelAll = np.arange((min(omega_K)),(max(omega_K)+10**6),2*np.pi*10**3)#Extend the range by 1MHz.
print("Number of mu is {0}".format(np.shape(muDelAll)))
#Add the additional term of phiM
Dimension_R = (const_Cou/(const_m*RF**2))**(1/3.) #Take back the dimension of r
#R0 = iterR.B_new[:,:,np.shape(iterR.B_new)[2]/2]*Dimension_R
R0 = B_new[:,:,np.shape(B_new)[2]/2]*Dimension_R
R1 = B_new[:,:,np.shape(B_new)[2]/2+1]*Dimension_R+B_new[:,:,np.shape(B_new)[2]/2-1]*Dimension_R
phi0 = R0z = np.sum(R0*kDel.reshape(1,-1),axis=1)#The first order phase
phi1 = R1z = np.sum(R1*kDel.reshape(1,-1),axis=1)

Bessel_n = Bessel(30,phi1)


#print Bessel_n
np.save("Bessel_n", Bessel_n)
print("Finished saving Bessel function")

nsegAll = [10,15,20]
ith = 1
jth = 4
phiM = phi0
fidAll = []
print("The micromotion phase is {0} and {1}".format(phi1[ith], phi1[jth]))
Detuning = 2*np.pi*10**2 #Add an additional detuning in case of mode-mu==0.
muDelAll = muDelAll + Detuning
np.save("muDelAll.out",muDelAll)


ind = range(indmax)
def main(ind, ith, jth, Bessel_n, nsegAll, muDelAll, phiM, phi1):
	numMu = np.shape(muDelAll)[0]
	for nseg in nsegAll:
	    #print nseg
	    time_seg = time.time()
	    indexlist = np.arange(ind,numMu,indmax)
	    print indexlist
	    for id in indexlist:
		if id%100 == 0:
		    print("nseg {0}, id {1}".format(nseg,id))
		mu = muDelAll[id]	
		#print(mu)
		time_loop = time.time()
		#A_k = gk[:,:,np.newaxis]*alpha*-1.0j
		alpha1_ki = alpha_Fun(nseg, tau, mu, ith, phiM[ith], omega_K, N, Bessel_n)
		alpha2_ki = alpha_Fun(nseg, tau, mu, jth, phiM[jth], omega_K, N, Bessel_n)
		A_k1 = gk[ith,:,np.newaxis]*alpha1_ki
		A_k2 = gk[jth,:,np.newaxis]*alpha2_ki
		betak = 1/(np.tanh((omega_K*const_hbar)/(2*const_k*Temperature))).reshape(-1)
		M = 0
		for k in range(np.shape(omega_K)[0]):
		    alpha_ki = A_k1[k,:].reshape(1,-1)
		    alpha_ki_dag = np.conjugate(alpha_ki.T)
		    alpha_kj =  A_k2[k,:].reshape(1,-1)
		    alpha_kj_dag = np.conjugate(alpha_kj.T)
		    M += betak[k]*(alpha_ki_dag.dot(alpha_ki)+alpha_kj_dag.dot(alpha_kj))

		M = np.real((M.T+M)/2)
		time0 = time.time()
		phi = np.imag(phifun(nseg, tau, mu, phiM[ith], phiM[jth], ith, jth, omega_K,Bessel_n))
		#args2 = zip(nseg, tau, mu, phiM[ith], phiM[jth], ith, jth, omega_K)
		#args2 = zip([nseg]*len(omega_K), [tau]*len(omega_K), [mu]*len(omega_K), [phiM[ith]]*len(omega_K), [phiM[jth]]*len(omega_K), [ith]*len(omega_K), [jth]*len(omega_K), omega_K, Bessel_n)
		#phi = pool.map(phifun, [nseg]*len(omega_K), [tau]*len(omega_K), [mu]*len(omega_K), [phiM[ith]]*len(omega_K), [phiM[jth]]*len(omega_K), [ith]*len(omega_K), [jth]*len(omega_K), omega_K, Bessel_n)
		
		#phi = pool.map(phifun, args2)

		Gamma = np.sum((gk[ith,:]*gk[jth,:]).reshape(-1,1,1)*phi,axis=0)
		Gamma = (Gamma.T+Gamma)/2.0

		time0 = time.time()
		(OmE, OmUnit) = scipy.linalg.eig(M,Gamma)
		OmUnit = OmUnit[:,-1].reshape(-1,1)
		Unit = np.dot(OmUnit.reshape(1,-1),np.dot(Gamma,OmUnit.reshape(-1,1)))
		#time0 = time.time()
		if Unit > 0:
		    Omega_i = np.sqrt(np.pi/(4*Unit))*OmUnit
		    Flag = 1
		else:
		    Omega_i = np.sqrt(-np.pi/(4*Unit))*OmUnit
		    Flag = -1

		fidelity = Avg_FID(Omega_i, tau, muDel=mu, alpha_ki=alpha1_ki, alpha_kj=alpha2_ki, Phi=phi, ith=ith,jth=jth, betak=betak, gk=gk, Flag=Flag)
		#print fidelity
		file = open("result/"+str(nseg)+"_fidelity_"+str(ind)+".out","a")
		#file.write(id)
		#file.write("\n")
		file.write(str(float(fidelity)))
		file.write("\n")
		file.write(str(float(mu)))
		file.write("\n")
		file.close()

		#print("fidelity here is {0}".format(fidelity))
		#time0 = time.time()
		#fidnsegAll.append(float(fidelity))
		#print("time in one loop is {0}".format(time.time()-time_loop))
	    print("time in one segment is {0}".format(time.time()-time_seg))
	    #fidAll.append(fidnsegAll)

pool = Pool(indmax)
ids = range(indmax)
func1 = partial(main, ith=ith, jth=jth, Bessel_n=Bessel_n, nsegAll=nsegAll, muDelAll=muDelAll, phiM=phiM, phi1=phi1)
pool.map(func1, ids)


	
