
import math
import numpy as np


class CorrelatedBrownianPathGenerator:
    
    def make_two_factor(rho,sigma_1,sigma_2):
        corr = np.array([[sigma_1**2, rho*sigma_1*sigma_2],[rho*sigma_1*sigma_2, sigma_2**2]])
        A = np.linalg.cholesky(corr)
        return CorrelatedBrownianPathGenerator(A)
    
    def make_one_factor(sigma=1.0):
        corr = np.array([[sigma**2]])
        A = np.linalg.cholesky(corr)
        return CorrelatedBrownianPathGenerator(A)
    
    def __init__(self, cholesky):
        self.cholesky = cholesky

        
    def Generate(self, T, N):
        d_t = T/N
        d_t_sqrt = math.sqrt(d_t)
        dim = self.cholesky.shape[0]
        
        d_W = np.zeros((dim,N))
        W = np.zeros((dim,N))
        for t in range(1,N):
            Z = np.random.normal(size=dim) 
            step_d_W = np.dot(self.cholesky,Z)*d_t_sqrt
            for j in range(dim):
                d_W[j,t] = step_d_W[j]
                W[j,t] = W[j,t-1] + step_d_W[j]
                
        class BrownianPath:
            def __init__(child):
                child.index = range(1,N)
                child.path_index = range(0,N)
                child.W = W
                child.d_t = [d_t] * N
                child.t = [d_t * idx for idx in range(N)]
                child.d_W = d_W
                
                
                
        return BrownianPath()









""" dX = \mu(X)dt + \sigma(X)dW """
""" dX = a X dt + b X dW """
class GeometricBrownianMotionKernel:
    def __init__(self, init=10.0, a=1.0, b=2.0):
        self.init = init
        self.a = a
        self.b = b
    def mu(self, t, x):
        return self.a * x
    def sigma(self, t, x):
        return self.b * x
    def d_sigma(self, t, x):
        return self.b
    
    def AnalyticValue(self, t : float, w : float):
        return self.init * math.exp(
            ( self.a - 0.5 * self.b**2 ) * t +
            self.b * w )

class AnalyticPathIntegrator:
    def __init__(self, kernel):
        self.kernel = kernel
    def IntegratePath(self, rf):
        path = []
        for idx in rf.path_index:
            path.append(
                self.kernel.AnalyticValue(
                    rf.t[idx], rf.W[0][idx]
                )
            )
        return path

class EulerMaruyamaPathIntegrator:
    def __init__(self, kernel):
        self.kernel = kernel
    def IntegratePath(self, rf):
        s = self.kernel.init 
        s_seq = [s]
        for idx in rf.index:
            t = rf.t[idx]
            d_s = 0.0
            d_s += self.kernel.mu(t,s)* rf.d_t[idx]
            d_s += self.kernel.sigma(t,s) * rf.d_W[0][idx]
            s += d_s
            s_seq.append(s)
        return s_seq
