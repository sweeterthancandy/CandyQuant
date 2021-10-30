
import math
import numpy as np


class RiskFactorProxy:
    def  __init__(self, parent, rf_index):
        self.parent = parent
        self.rf_index = rf_index

    @property
    def index(self): return self.parent.index
    
    @property
    def d_index(self): return self.parent.d_index
    
    @property
    def t(self): return self.parent.t
    
    @property
    def d_t(self): return self.parent.d_t
    
    @property
    def W(self): return self.parent.W[self.rf_index]
    
    @property
    def d_W(self): return self.parent.d_W[self.rf_index]
    


        
class CorrelatedBrownianPath:
    def __init__(self, index, d_index, t, d_t, W, d_W):
        self.index = index
        self.d_index = d_index
        self.t = t
        self.W = W
        self.d_t = d_t
        self.d_W = d_W


    def risk_factor(self, index):
        return RiskFactorProxy(self, index)


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
        corr = np.array(corr_mtx)
        A = np.linalg.cholesky(corr)
        return CorrelatedBrownianPathGenerator(A)
    
    def __init__(self, cholesky):
        self.cholesky = cholesky
        
    def generate(self, T, N):
        assert N >= 1
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
        """

        Args:
            T : The timeline to generate a path for, ie TL = [0,T]
            N : The number of d_W steps => have N + 1 W_t steps to include W_0 = 0


        
        """
        assert N >= 1
        d_t = T/(N)
        d_t_sqrt = math.sqrt(d_t)

        dim = self.cholesky.shape[0]

        d_W = self.generate_dw_matrix(d_t_sqrt, N)

        assert d_W.shape == (dim,N)

        aux = []
        for j in range(dim):
            aux.append(np.concatenate(([0],np.cumsum(d_W[j]))))
        W = np.array(aux)



        result = CorrelatedBrownianPath(
            index= range(N+1),
            d_index=range(N),
            t = np.linspace(0,T,N+1),
            d_t = [d_t]*N,
            W=W,
            d_W=d_W)


        

        assert result.d_W.shape == (dim,N)
        assert len(result.d_t) == N
        
        assert result.W.shape == (dim,N+1)
        assert len(result.t) == N+1
        

        return result










""" dX = \mu(X)dt + \sigma(X)dW """
""" dX = a X dt + b X dW """
class GeometricBrownianMotionKernel:
    def __init__(self, init=10.0, a=0.02, b=0.2):
        self.init = init
        self.a = a
        self.b = b
    def mu(self, t, x):
        return self.a * x
    def sigma(self, t, x):
        return self.b * x
    def d_sigma(self, t, x):
        return self.b
    
    def analytic_integral(self, t : float, w : float):
        return self.init * math.exp(
            ( self.a - 0.5 * self.b**2 ) * t +
            self.b * w )

class AnalyticPathIntegrator:
    def __init__(self, kernel):
        self.kernel = kernel
    def integrate(self, rf):
        path = []
        for idx in rf.index:
            path.append(
                self.kernel.analytic_integral(
                    rf.t[idx], rf.W[idx]
                )
            )
        return path

class EulerMaruyamaPathIntegrator:
    def __init__(self, kernel):
        self.kernel = kernel
    def integrate(self, rf):
        s = self.kernel.init 
        s_seq = [s]
        for idx in rf.d_index:
            t = rf.t[idx]
            d_s = 0.0
            d_s += self.kernel.mu(t,s)* rf.d_t[idx]
            d_s += self.kernel.sigma(t,s) * rf.d_W[idx]
            s += d_s
            s_seq.append(s)

        assert len(s_seq) == len(rf.W)
        return np.array(s_seq)





def call_option_mc(x,K,tau,r,sigma,num_paths=32000,how='analytic-path'):
    brownian_motion_gen = CorrelatedBrownianPathGenerator.make_one_factor()
    kernel = GeometricBrownianMotionKernel(init=x, a=r, b=sigma)

    if how == 'analytic-path': 
        PATHS_PER_SIM = 1
        proc_int = AnalyticPathIntegrator(kernel)
    elif how == 'integrate-path':
        PATHS_PER_SIM = int(math.sqrt(num_paths)/10)
        proc_int = EulerMaruyamaPathIntegrator(kernel)
    else:
        raise RuntimeError("dont know how={}".format(how))

    # only need the final value

    path_payoffs = []
    for j in range(num_paths):
        rf = brownian_motion_gen.generate(tau,PATHS_PER_SIM).risk_factor(0)
        proc_path = proc_int.integrate(rf)
        terminal_value = proc_path[-1]
        payoff = max(terminal_value - K, 0)
        path_payoffs.append(payoff)

    V_T = np.mean(path_payoffs)
    discounted_V_T = math.exp(-r * tau ) * V_T
    return discounted_V_T
