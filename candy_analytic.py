import math

class call_option_params:
    """
    Class to encode the parameters to black schedole call option
    """
    def __init__(self, time_to_expiry=1.0, inital_value=50.0, strike=60.0, vol=0.2, r=0.04):
        self.time_to_expiry = time_to_expiry
        self.inital_value   = inital_value
        self.strike         = strike
        self.vol            = vol
        self.r              = r
        
def call_option_analytic(x,K,tau,r,sigma):
    
    if tau < 0.0:
        return max(0.0,x - K)
    
    from scipy.stats import norm

    tmp0 = math.log(x/K)
    tmp1 = sigma**2/2
    
    d_plus = 1.0/sigma/math.sqrt(tau)*(tmp0 + ( r + tmp1 )*tau )
    d_minus = 1.0/sigma/math.sqrt(tau)*(tmp0 + ( r - tmp1 )*tau )
    
    return x * norm.cdf( d_plus ) - K * math.exp(-r*tau)* norm.cdf( d_minus )

def call_option_analytic_fp(p):
    return call_option_analytic(p.inital_value,
                                p.strike,
                                p.time_to_expiry,
                                p.r,
                                p.vol)


def digital_call_option_analytic(x,K,tau,r,sigma):
    from scipy.stats import norm
    tmp0 = math.log(x/K) + ( r - sigma**2/2)*tau
    tmp1 = tmp0 / sigma / math.sqrt(tau)
    tmp2 = norm.cdf(tmp1)
    tmp3 = tmp2 * math.exp( - r * tau)
    return tmp3

def digital_call_option_analytic_fp(p):
    return digital_call_option_analytic(p.inital_value,
                                p.strike,
                                p.time_to_expiry,
                                p.r,
                                p.vol)

