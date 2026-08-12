import torch
from model import sample_checkerboard, checkerboard_density

x = sample_checkerboard(2000)
density = checkerboard_density(x)
print(f"Percentage of points strictly inside checkerboard mask: {(density > 0.5).mean() * 100:.2f}%")
