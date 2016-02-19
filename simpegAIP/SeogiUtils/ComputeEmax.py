import numpy as np
from SimPEG import Utils

def computePeta(time, eEM, ja, sigma, mesh):

	ntime = time.size
	eEM_CC = mesh.aveE2CCV*eEM
	eabs = (eEM_CC)**2
	Eabs = np.zeros((mesh.nC, ntime))
	for i in range(ntime):
		Eabs[:,i] = (np.reshape(eabs[:,i], (mesh.nC, 3), order = 'F')).sum(axis = 1)
	indEmax = np.argmax(Eabs, axis = 1)
	indEmax_e = (mesh.aveE2CC.T*indEmax).astype(int)
	IndEmax_e = Utils.sub2ind((eEM).shape, np.c_[np.arange((eEM).shape[0]), indEmax_e])
	eEM_max = Utils.mkvc(eEM)[IndEmax_e]
	eEM_max_CC = mesh.aveE2CCV*eEM_max
    	
	EEMIPdotErefFund = np.zeros((mesh.nC, ntime))
	
	for i in range(ntime):
		EEMIPdotErefFund[:,i] = (np.reshape(eEM_CC[:,i], (mesh.nC, 3), order = 'F')*(eEM_max_CC).reshape((mesh.nC, 3), order='F')).sum(axis=1)

	peta = np.zeros((mesh.nC, ntime))
	Eabsmax = (eEM_max_CC**2).reshape((mesh.nC, 3), order='F').sum(axis=1)
	weFund = Utils.sdiag(1./Eabsmax)*EEMIPdotErefFund
	
	ja_CC = mesh.aveE2CCV*ja

	for i in range(ntime):
	    dum = Utils.sdiag(1./Eabsmax)*(np.reshape(ja_CC[:,i], (mesh.nC, 3), order = 'F')*(eEM_max_CC).reshape((mesh.nC, 3), order='F')).sum(axis=1)    
	    dum_pos=dum.copy()
	    dum_neg=dum.copy()
	    dum_neg[dum>0.] = 0.
	    peta[:,i] = -dum_neg/sigma

	return peta
