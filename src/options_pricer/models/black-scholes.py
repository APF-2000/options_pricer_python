import numpy as np
from scipy.stats import norm
def d_1(sigma: float,T: float,t: float,S: float,K: float,r: float) -> float:
    return 1/(sigma*np.sqrt(T-t))*(np.log(S/K) + (r + (sigma**2)/2)*(T-t))

def d_2(sigma: float,T: float,t: float,S: float,K: float,r: float) -> float:
    return d_1(sigma,T,t,S,K,r) - sigma*np.sqrt(T-t)

def european_call_option(sigma: float,T: float,t: float,S: float,K: float,r: float):
    return norm.cdf(d_1(sigma,T,t,S,K,r))*S - norm.cdf(d_2(sigma,T,t,S,K,r))*K*np.exp(-r*(T-t))

def european_put_option(sigma: float,T: float,t: float,S: float,K: float,r: float):
    return K*norm.cdf(-d_2(sigma,T,t,S,K,r))*np.exp(-r*(T-t)) - norm.cdf(-d_1(sigma,T,t,S,K,r))*S

S = 100      # spot price
K = 100      # strike price
T = 1.0      # maturity in years
t = 0.0      # current time
r = 0.05     # risk-free rate (5%)
sigma = 0.2  # volatility (20%)

call = european_call_option(sigma, T, t, S, K, r)
put = european_put_option(sigma, T, t, S, K, r)

print(f"Call price: {call:.4f}")
print(f"Put price:  {put:.4f}")