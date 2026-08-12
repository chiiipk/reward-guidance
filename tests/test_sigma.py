import sys
sys.path.append('.')
import numpy as np
from pytest import approx
from checkerboard.sample import _posterior_var

def test_posterior_var_endpoints():
    assert _posterior_var(0.0, 1.73) == approx(1.73**2)   # t=0: noise, chưa biết gì
    assert _posterior_var(1.0, 1.73) == approx(0.0)       # t=1: data, biết chính xác
    # đơn điệu giảm
    ts = np.linspace(0, 1, 50)
    v = [_posterior_var(t, 1.73) for t in ts]
    assert all(np.diff(v) <= 1e-9)

if __name__ == '__main__':
    test_posterior_var_endpoints()
    print("test_posterior_var_endpoints passed")
