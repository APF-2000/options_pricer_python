import math
import pytest
import numpy as np
from .black_scholes import european_call_option, european_put_option

# Reference parameters
S = 100.0   # Spot price
K = 100.0   # Strike price
T = 1.0     # Maturity (years)
t = 0.0     # Current time
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

# Expected benchmark values (from textbook or verified calculator)
EXPECTED_CALL = 10.4506
EXPECTED_PUT  = 5.5735
TOL = 1e-4

def test_call_price_matches_expected():
    """Black–Scholes call should match reference value."""
    price = european_call_option(sigma, T, t, S, K, r)
    assert math.isclose(price, EXPECTED_CALL, rel_tol=0, abs_tol=TOL)

def test_put_price_matches_expected():
    """Black–Scholes put should match reference value."""
    price = european_put_option(sigma, T, t, S, K, r)
    assert math.isclose(price, EXPECTED_PUT, rel_tol=0, abs_tol=TOL)

def test_put_call_parity():
    """Check that call and put satisfy C - P = S e^{-qT} - K e^{-rT} (q=0)."""
    C = european_call_option(sigma, T, t, S, K, r)
    P = european_put_option(sigma, T, t, S, K, r)
    lhs = C - P
    rhs = S - K * math.exp(-r * (T - t))
    assert abs(lhs - rhs) < 1e-6

@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.5])
def test_monotonicity_in_volatility(sigma):
    """Call price should increase with volatility."""
    price_low = european_call_option(0.1, T, t, S, K, r)
    price_high = european_call_option(0.5, T, t, S, K, r)
    assert price_high > price_low

def test_price_limits_at_expiry():
    """At expiry (T=t), option value = intrinsic value."""
    S = 120
    K = 100
    T = t = 1.0  # same time
    sigma = 0.2
    r = 0.05
    call = european_call_option(sigma, T, t, S, K, r)
    put  = european_put_option(sigma, T, t, S, K, r)
    assert call == max(S - K, 0)
    assert put == max(K - S, 0)

def test_zero_volatility_case():
    """When sigma → 0, price = discounted intrinsic under risk-free growth."""
    sigma = 1e-12
    price = european_call_option(sigma, T, t, S, K, r)
    expected = max(S * math.exp(r*(T-t)) - K, 0) * math.exp(-r*(T-t))
    assert abs(price - expected) < 1e-6
