from SimPEG.EM.Base import BaseEMProblem, EMPropMap
from SimPEG.Maps import IdentityMap
import numpy as np

class BaseAEMProblem(BaseEMProblem):

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    @property
    def MeS(self):
        """
            Current term: MeS = Me(js)
        """        
        if getattr(self, '_MeS', None) is None:
            self._MeS = np.zeros((self.mesh.nE,len(self.survey.srcList)))
            for isrc, src in enumerate(self.survey.srcList):       
                self._MeS[:,isrc] = src.getMeS(self.mesh, self.MfMui)
        return self._MeS

class ColeColeMap(IdentityMap):
    """
        Takes a vector of [sigmaInf, eta, tau, c]
        and sets 

            - sigmaInf: Conductivity at infinite frequency (S/m)
            - eta: Chargeability (V/V)
            - tau: time constant (sec)
            - c: Frequency dependency (Dimensionless)            
    """
    sigmaInf = None
    eta = None
    tau = None
    c = None

    def __init__(self, mesh, nC=None):
        self.mesh = mesh
        self.nC = nC or mesh.nC

    def _transform(self, m):        
        m = m.reshape((self.mesh.nC, 4), order = "F")
        self.sigmaInf = m[:,0]
        self.eta = m[:,1]
        self.tau = m[:,2]
        self.c = m[:,3]
        return Utils.mkvc(m)
    
    @property
    def shape(self):
        return (self.nP, self.nP)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.nC*4       
