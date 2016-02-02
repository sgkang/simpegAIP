from simpegAIP.TD import BaseATEMProblem
from SimPEG.Utils import mkvc, sdiag
import numpy as np
from SimPEG.EM.TDEM.BaseTDEM import FieldsTDEM
from SimPEG.EM.TDEM.SurveyTDEM import SurveyTDEM

class FieldsATEMIP_store_bej(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'b': 'F','e': 'E','j': 'E'}

    
    def startup(self):
        self.MeI  = self.survey.prob.MeI
        self.MeS = self.survey.prob.MeS
        self.edgeCurlT = self.survey.prob.mesh.edgeCurl.T
        self.MfMui     = self.survey.prob.MfMui

class ProblemATEMIP_b(BaseATEMIPProblem):

    """
        Time-Domain EM-IP problem - B-formulation
        TDEM_b treats the following discretization of Maxwell's equations

    """

    store_fields = False # Only store magnetic field
    sigmaHatDict = {}

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseATEMProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    solType = 'b' #: Type of the solution, in this case the 'b' field
    surveyPair = SurveyTDEM
    _FieldsForward_pair = FieldsATEMIP_store_bej     #: used for the forward calculation only

    ####################################################
    # Internal Methods
    ####################################################

    
    @property
    def MeSigmaInf(self):
    	if getattr(self, '_MeSigmaInf', None) is None:
            self._MeSigmaInf = self.mesh.getEdgeInnerProduct(self.sigmaInf)
        return self._MeSigmaInf

    @property
    def MeI(self):
        if getattr(self, '_MeI', None) is None:
            self._MeI = mesh.getEdgeInnerProduct(invMat=True)
        return self._MeI
    
    def MeAc(self, dt):
        gamma = self.getGamma(dt)
        val = self.sigmaInf - gamma
        return self.mesh.getEdgeInnerProduct(val)                    

    MeAcI = lambda self, dt: sdInv(self.MeAc(dt))

    def MeCnk(self, n, k):
        tn = self.times(n)
        tk = self.times(k)
        val = self.sigmaHat(tn-tk)
        return self.mesh.getEdgeInnerProduct(val)

    _nJpLast = None
    _JpLast = None

    #This needs to go somewhere ...
    # def get_e(self, tInd):
    #     e = np.zeros((self.mesh.nE,len(self.survey.srcList)))        
    #     for isrc, src in enumerate(self.survey.srcList):       
    #         e[:,isrc] = F[src,'e',tInd].flatten()
    # 	return e

    # def get_b(self, tInd):
    #     b = np.zeros((self.mesh.nF,len(self.survey.srcList)))        
    #     for isrc, src in enumerate(self.survey.srcList):       
    #         b[:,isrc] = F[src,'b',tInd].flatten()
    # 	return b    	

    def getJp(self, tInd):
    	"""
    		Computation of polarization currents
    	"""
    	# Not sure ... why 
        if tInd == self._nJpLast:
        	print ">> Curious when self._nJpLast is used:", self._nJpLast
            jp = self._JpLast
        else:
            dt = self.timeSteps(tInd)
            kappa = self.getKappa(dt)
            MeK = self.mesh.getEdgeInnerProduct(kappa)
            jp = MeK*self.get_e(tInd)            
            for k in range(tInd):
                dt = self.timeSteps(k)
                jp += (dt/2)*self.MeCnk(tInd+1,k)*self.F[:,'e',k+1]
                jp += (dt/2)*self.MeCnk(tInd+1,k+1)*self.F[:,'e',k+1]
            self._nJpLast = tInd
            self._JpLast = jp
        return jp

    def getA(self, tInd):
        """
            :param int tInd: Time index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        dt = self.timeSteps[tInd]
        return self.MfMui*self.mesh.edgeCurl*self.MeAcI(dt)*self.mesh.edgeCurl.T*self.MfMui + self.MfMui*(1.0/dt)

    def getRHS(self, tInd, F):
        dt = self.timeSteps[tInd]        
        B_last = self.F[:,'b',tInd]
        RHS = self.MfMui*(1/dt)*B_last \
            - self.MfMui*self.mesh.edgeCurl*self.MeAcI(dt)*self.getJp(tInd) 
        if self.waveformType != "STEPOFF":
            RHS += self.MfMui*self.mesh.edgeCurl*self.MeAcI(dt)*self.MeS*self.current[tInd+1]
        return RHS

    def updateFields(self, bn, tInd):
        dt = self.timeSteps[tInd]
        jp = self.getJp(tInd)
        en = self.MeAcI(dt)*self.mesh.edgeCurl.T*self.MfMui*bn \
           + self.MeAcI(dt)*jp
        if self.waveformType != "STEPOFF":
            en -= self.MeAcI(dt)*self.MeS*self.current[tInd+1]
        jn = self.MeI*(self.MeAc(dt)*en - jp)
        return en, jn

    def getKappa(self, dt):
        setc = np.c_[self.sigmaInf, self.eta, self.tau, self.c]
        unqEtc, uETCind, invInd = uniqueRows(setc)
        kappaVals = []
        for (sv, ev, tv, cv) in unqEtc:
            if ev == 0:
                kappaVal = 0.
            elif cv == 1.:
                kappaVal = (dt/2.)*sv*ev/((1. - ev)*tv)
            else:
                m, d = self.getMD(dt, sv, ev, tv, cv)
                kappaVal = (m*dt**cv)/(cv+1) + d*dt/2
            kappaVals.append(kappaVal)
        kappa = np.array(kappaVals)[invInd]
        return kappa

    def getGamma(self, dt):
        setc = np.c_[self.sigmaInf, self.eta, self.tau, self.c]
        unqEtc, uETCind, invInd = uniqueRows(setc)
        gammaVals = []
        for (sv, ev, tv, cv) in unqEtc:
            if ev == 0:
                gammaVal = 0.
            elif cv == 1.:
                gammaVal = (dt/2.)*sv*ev/((1. - ev)*tv)
            else:
                m, d = self.getMD(dt, sv, ev, tv, cv)
                gammaVal = (m*dt**cv)/(cv*(cv+1)) + d*dt/2.
            gammaVals.append(gammaVal)
        gamma = np.array(gammaVals)[invInd]
        return gamma

    def getMD(self, dt, sv, ev, tv, cv):

        def CCTF(t, sigmaInf, eta, tau, c):
            F = lambda frq: (1j*2*np.pi*frq)*ColeCole(frq, sigmaInf=sigmaInf, eta=eta, tau=tau, c=c)
            return transFilt(F,t)

        t = np.r_[1e-20, dt]


        y = CCTF(t, sv, ev, tv, cv)
        x = t**(cv-1)
        m = (y[0]-y[1])/(x[0]-x[1])
        d = y[1] - m*x[1]
        return m, d
