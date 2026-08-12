import re

with open("smoke_test_second_order.py", "r") as f:
    code = f.read()

code = code.replace("for lam in [1e-1, 1e-2, 1e-3]:", "for lam in [1e-2, 1e-3, 1e-4]:")

with open("smoke_test_second_order.py", "w") as f:
    f.write(code)
