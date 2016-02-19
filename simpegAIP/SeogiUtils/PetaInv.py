from SimPEG import Problem, Survey, Utils, Maps
from SeogiUtils import Convolution
import numpy as np

class PetaInvProblem(Problem.BaseProblem):
    surveyPair = Survey.BaseSurvey
    P = None
    J = None
    time = None
    we = None
    
    def __init__(self, mesh, mapping, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, mapping, **kwargs)
        
    def fields(self, m, u=None):
        mtemp=self.mapping*m
        self.J = petaJconvfun(m[0],m[1],self.we,self.time,self.P)
        return petaconvfun(mtemp[0],mtemp[1], self.we, self.time, self.P)

    def Jvec(self, m, v, u=None):
#         J = petaJconvfun(m[0],m[1],self.we,self.time,self.P)
        P = self.mapping.deriv(m)
        jvec = self.J.dot(P*v)
        return jvec

    def Jtvec(self, m, v, u=None):
#         J = petaJconvfun(m[0],m[1],self.we,self.time,self.P)
        P = self.mapping.deriv(m)
        jtvec =P.T*(self.J.T.dot(v))
        return jtvec
    
def petaconvfun(a, b, we, time, P):    
    kernel = lambda x: a*np.exp(-b*x)
    temp = kernel(time)
    temp = Convolution.CausalConvIntSingle(we, time, kernel)
    out = P*temp
    return out

def petaJconvfun(a, b, we, time, P):    
    kerneleta = lambda x: np.exp(-b*x)
    kerneltau = lambda x: -a*x*np.exp(-b*x)
    tempeta = kerneleta(time)
    temptau = kerneltau(time)
    tempeta = Convolution.CausalConvIntSingle(we, time, kerneleta)
    temptau = Convolution.CausalConvIntSingle(we, time, kerneltau)
    J = np.c_[P*tempeta, P*temptau]
    return J    

class PetaSurvey(Survey.BaseSurvey):

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)
        
    @Utils.requires('prob')
    def dpred(self, m, u=None):
        return self.prob.fields(m)     
    
    def residual(self, m, u=None):
        if self.dobs.size ==1:
            return Utils.mkvc(np.r_[self.dpred(m, u=u) - self.dobs])
        else:
            return Utils.mkvc(self.dpred(m, u=u) - self.dobs)
    