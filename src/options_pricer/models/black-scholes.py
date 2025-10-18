import numpy as np
from scipy.stats import norm
def d_1(sigma: float,T: float,t: float,S: float,K: float,r: float) -> float:
    return 1/(sigma*np.sqrt(T-t))*(np.log(S/K) + (r + (sigma**2)/2)*(T-t))

def d_2(sigma: float,T: float,t: float,S: float,K: float,r: float) -> float:
    return d_1(sigma,T,t,S,K,r) - sigma*np.sqrt(T-t)

def european_call_option(sigma: float,T: float,t: float,S: float,K: float,r: float):
    return norm.cdf(d_1(sigma,T,t,S,K,r))*S - norm.cdf(d_2(sigma,T,t,S,K,r))*K*np.exp(-r*(T-t))

def european_put_option(sigma: float,T: float,t: float,S: float,K: float,r: float):
    return -K*norm.cdf(d_2(sigma,T,t,S,K,r))*np.exp(-r*(T-t)) - norm.cdf(d_1(sigma,T,t,S,K,r))*S
