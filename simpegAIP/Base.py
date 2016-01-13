from SimPEG.EM.Base import BaseEMProblem, EMPropMap
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
