
import math
import numpy as np


        
class CorrelatedBrownianPath:
    def __init__(self, index, path_index, t, d_t, W, d_W):
        self.index = index
        self.path_index = path_index
        self.t = t
        self.d_t = d_t
        self.W = W
        self.d_W = d_W

class CorrelatedBrownianPathGenerator:
    """
    This generates correlation browniam motion paths
    """
    
    def make_two_factor(rho,sigma_1,sigma_2):
        corr_mtx = [
                [sigma_1**2, rho*sigma_1*sigma_2],
                [rho*sigma_1*sigma_2, sigma_2**2]
        ]
        corr = np.array(corr_mtx)
        A = np.linalg.cholesky(corr)
        return CorrelatedBrownianPathGenerator(A)
    
    def make_one_factor(sigma=1.0):
        corr_mtx = [[sigma**2]]
        corr = np.array()
        A = np.linalg.cholesky(corr)
        return CorrelatedBrownianPathGenerator(A)
    
    def __init__(self, cholesky):
        self.cholesky = cholesky
        
    def generate(self, T, N):
        assert N >= 2
        d_t = T/(N-1)
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
                child.t = np.linspace(0,T,N)
                child.d_W = d_W
                
                
        return BrownianPath()


    def generate_dw_matrix(self, d_t_sqrt, size):
        """
        This is the important step, generates a dim x size matrix of correlated dw
        """
        dim = self.cholesky.shape[0]
        
        d_W = np.zeros((dim,size))
        for index in range(size):
            Z = np.random.normal(size=dim) 
            step_d_W = np.dot(self.cholesky,Z)*d_t_sqrt
            d_W[:,index] = step_d_W
        return d_W
    def generate_dw_matrix(self, d_t_sqrt, size):
        """
        This is the important step, generates a dim x size matrix of correlated dw
        """
        dim = self.cholesky.shape[0]
        
        aux = []
        for index in range(size):
            Z = np.random.normal(size=dim) 
            step_d_W = np.dot(self.cholesky,Z)*d_t_sqrt
            aux.append(step_d_W)
        return np.array(aux).transpose()

    

    def generate(self, T, N):
        assert N >= 2
        d_t = T/(N-1)
        d_t_sqrt = math.sqrt(d_t)

        dim = self.cholesky.shape[0]

        d_W = self.generate_dw_matrix(d_t_sqrt, N-1)

        assert d_W.shape == (dim,N-1)

        aux = []
        for j in range(dim):
            aux.append(np.concatenate(([0],np.cumsum(d_W[j]))))
        W = np.array(aux)



        return CorrelatedBrownianPath(
            index= range(1,N),
            path_index=range(0,N),
            t = np.linspace(0,T,N),
            d_t = [d_t]*N,
            W=W,
            d_W=d_W)










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
