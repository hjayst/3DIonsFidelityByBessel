import numpy as np
class mode_config():
	gamma0 = 1
	step = 30000 # Cooling steps
	T = 1200*np.pi   #Cooling time
	step2 = 10000 
	T2 = 10*np.pi
	repeat = False
	alpha = 1
	m = 5
	m2 = 2
	num = 10000 #For ions numbers around 50, num should be 3000
	

"""	
class constants():
    const_hbar=1.0545718*10**(-34);
    const_e=1.60217662*10**(-19)
    const_mp=1.6605390*10**-27
    const_m=const_mp*171.0
    const_eps=8.854187818*10**(-12)
    const_Cou=const_e**2/(4*np.pi*const_eps)
    const_k=1.38064852*10**(-23)
"""
const_hbar=1.0545718*10**(-34);
const_e=1.60217662*10**(-19)
const_mp=1.6605390*10**-27
const_m=const_mp*171.0
const_eps=8.854187818*10**(-12)
const_Cou=const_e**2/(4*np.pi*const_eps)
const_k=1.38064852*10**(-23)

NDel=1; # one ion on each end discarded
l0=40*10**(-6); #length unit l0
omx=2*np.pi*3*10**6; # OmxCM (transverse direction)
gamma4=4.3; # optimized to minimize relative standard derivation
wavelength=355*10**(-9); #wavelength of each laser
kDel=2*2*np.pi/wavelength; #counter-propagating Raman beams, Delta k = 2k 
Temperature=const_hbar*omx/const_k; # 0.5 phonon per mode

tau=300*10**(-6)
N=50
RF=40*10**6*2*np.pi

 
