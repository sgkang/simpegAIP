from SimPEG import *
import simpegEM as EM
import matplotlib.pyplot as plt
from scipy.constants import mu_0
import SeogiUtils as SeUtils

# Step1: set mesh and model
mesh = Utils.meshutils.readUBCTensorMesh('../mesh.msh')
# sigma = Utils.meshutils.readUBCTensorModel('../sigmaINF.con', mesh)
sigma = np.load('../sigmaInf.npy')
xr =  np.arange(-300., 301., 40.)
yr =  np.arange(-300., 301., 100.)
xyz_tx = Utils.ndgrid(xr, yr, np.r_[30.])
ntx = xyz_tx.shape[0]

indz = 18
indy = 43

print 'z = ', mesh.vectorCCz[indz]
print 'y = ', mesh.vectorCCy[indy]

fig, ax = plt.subplots(1,2, figsize = (12, 5))
mesh.plotSlice(np.log10(sigma), vType='CC', normal='Y', grid=True, ax = ax[0], clim = (-4, -1), ind = indy)
mesh.plotSlice(np.log10(sigma), vType='CC', normal='Z', grid=True, ax = ax[1], clim = (-4, -1), ind = indz)
ax[1].plot(xyz_tx[:,0], xyz_tx[:,1], 'k.', ms = 15)
ax[0].set_xlim(-500, 500)
ax[0].set_ylim(-500, 500)
ax[1].set_xlim(-500, 500)
ax[1].set_ylim(-500, 500)
plt.show()


# Step2: Set mapping
active = mesh.gridCC[:,2] < 0.
actMap = Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nC)
mapping = Maps.ExpMap(mesh) * actMap
model = np.log(sigma[active])

ntx = 1
# Step3: Set Survey and Problem
tx_all = []
rx_all = []
time = np.logspace(-5.5, -5, 10)
for i in range (ntx):
    rx_temp = EM.TDEM.RxTDEM(xyz_tx[i,:], time, 'bz')
    tx_temp = EM.TDEM.TxTDEM(np.array([xyz_tx[i,:]]), 'VMD_MVP', [rx_temp])    
    rx_all.append(rx_temp)
    tx_all.append(tx_temp)


survey = EM.TDEM.SurveyTDEM(tx_all)
prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping,  verbose=True)
prb.Solver = Utils.SolverUtils.SolverWrapD(sp.linalg.splu, factorize=True)
prb.timeSteps = [(1e-06, 10), (1e-05, 10)]
# prb.timeSteps = [(1e-06, 10), (1e-05, 10), (1e-04, 10)]
if prb.ispaired:
    prb.unpair()
if survey.ispaired:
    survey.unpair()
prb.pair(survey)

# Step4: Compute fields
F = prb.fields(model)

# Step5: Compute jmax

ntime = prb.times.size
j_max  = []
e_max_abs = []
for i in range(ntx):
    eEM = F[tx_all[i], 'e', [np.arange(ntime)]]
    eEM_CC = mesh.aveE2CCV*eEM
    eabs = (eEM_CC)**2
    Eabs = np.zeros((mesh.nC, ntime))
    for i in range(ntime):    
        Eabs[:,i] = (np.reshape(eabs[:,i], (mesh.nC, 3), order = 'F')).sum(axis = 1)
    indEmax = np.argmax(Eabs, axis = 1)
    indEmax = np.r_[indEmax, indEmax, indEmax]
    IndEmax_CC = Utils.sub2ind((mesh.nC*3, ntime), np.c_[np.arange((eEM_CC).shape[0]), indEmax])
    eEM_max_CC = Utils.mkvc(eEM_CC)[IndEmax_CC].reshape((mesh.nC, 3), order = 'F')    
    e_max_abs.append((eEM_max_CC**2).sum(axis = 1))
    jx_max_temp = Utils.sdiag((Utils.sdiag(sigma)*eEM_max_CC)[:,0])
    jy_max_temp = Utils.sdiag((Utils.sdiag(sigma)*eEM_max_CC)[:,1])
    jz_max_temp = Utils.sdiag((Utils.sdiag(sigma)*eEM_max_CC)[:,2])    
    j_max.append( sp.vstack((jx_max_temp, jy_max_temp, jz_max_temp)) )	

# Step6: Save jmax
np.save('e_max_abs', e_max_abs)
np.save('j_max', j_max)
