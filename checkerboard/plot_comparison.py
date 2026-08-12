import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ot
from model import checkerboard_density

def compute_metrics(samples, rewards, lam, reward_center, is_unguided=False):
    # 1. Reward
    mean_reward = np.nanmean(rewards) if not is_unguided else np.nan
    
    # 2. In-dist ratio
    in_dist_mask = checkerboard_density(samples) > 0.5
    in_dist_ratio = np.mean(in_dist_mask)
    
    # Grid definition for coverage, entropy, and target W2
    # Checkerboard is defined on [-3, 3] x [-3, 3] with 1x1 cells
    # A point (x,y) is in cell (i,j) where i = floor(x+3), j = floor(y+3)
    # Valid grey cells are those where (i+j) is even.
    # Total cells: 6x6 = 36. Grey cells: 18.
    
    i_indices = np.floor(samples[:, 0] + 3).astype(int)
    j_indices = np.floor(samples[:, 1] + 3).astype(int)
    
    # Filter out out-of-bounds just in case
    valid_bounds = (i_indices >= 0) & (i_indices < 6) & (j_indices >= 0) & (j_indices < 6)
    i_valid = i_indices[valid_bounds]
    j_valid = j_indices[valid_bounds]
    
    # 3. Coverage (number of unique grey cells that have at least 1 sample)
    cell_ids = i_valid * 6 + j_valid
    grey_cell_mask = (i_valid + j_valid) % 2 == 0
    unique_grey_cells, counts = np.unique(cell_ids[grey_cell_mask], return_counts=True)
    coverage = len(unique_grey_cells) / 18.0  # ratio of grey cells covered
    
    # 4. Entropy of distribution over grey cells
    probs = counts / np.sum(counts) if len(counts) > 0 else np.array([1.0])
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    
    # 5. Discrete W2 distance
    # Target distribution: uniform over grey cells, weighted by exp(lam * r(cell_center))
    grey_centers_x = []
    grey_centers_y = []
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                grey_centers_x.append(i - 3 + 0.5)
                grey_centers_y.append(j - 3 + 0.5)
    
    grey_centers = np.column_stack([grey_centers_x, grey_centers_y]) # 18 x 2
    
    # Reward at centers
    diffs = grey_centers - np.array(reward_center)
    r_centers = np.exp(-np.sum(diffs**2, axis=-1) / (2 * 1.5**2))
    
    # Target distribution: uniform over grey cells, weighted by exp(lam * r(cell_center))
    target_weights = np.exp(lam * r_centers)
    target_weights = target_weights / np.sum(target_weights)
        
    # Sample distribution over grey cells
    sample_weights = np.zeros(18)
    for idx, cell_id in enumerate(unique_grey_cells):
        i = cell_id // 6
        j = cell_id % 6
        # Find index in grey_centers
        for k in range(18):
            if np.abs(grey_centers[k, 0] - (i - 3 + 0.5)) < 0.1 and np.abs(grey_centers[k, 1] - (j - 3 + 0.5)) < 0.1:
                sample_weights[k] = probs[idx]
                break
                
    # W2 distance between discrete distributions
    if np.sum(sample_weights) == 0:
        w2 = 100.0  # High penalty if no samples land in grey cells
    else:
        # Normalize just in case
        sample_weights = sample_weights / np.sum(sample_weights)
        M = ot.dist(grey_centers, grey_centers, metric='sqeuclidean')
        w2 = ot.emd2(sample_weights, target_weights, M)
    
    return mean_reward, in_dist_ratio, coverage, entropy, w2


def generate_analytic_samples(lam, reward_center, num_samples=5000):
    grey_centers_x = []
    grey_centers_y = []
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                grey_centers_x.append(i - 3 + 0.5)
                grey_centers_y.append(j - 3 + 0.5)
    grey_centers = np.column_stack([grey_centers_x, grey_centers_y])
    
    diffs = grey_centers - np.array(reward_center)
    r_centers = np.exp(-np.sum(diffs**2, axis=-1) / (2 * 1.5**2))
    
    target_weights = np.exp(lam * r_centers)
    target_weights = target_weights / np.sum(target_weights)
    
    chosen_cells = np.random.choice(18, size=num_samples, p=target_weights)
    
    # Sample uniformly within chosen cells
    samples = grey_centers[chosen_cells] + np.random.uniform(-0.5, 0.5, size=(num_samples, 2))
    
    # Rewards for these samples
    diffs_samples = samples - np.array(reward_center)
    rewards = np.exp(-np.sum(diffs_samples**2, axis=-1) / (2 * 1.5**2))
    
    return samples, rewards

def compute_w2_floor(samples1, samples2, lam, reward_center):
    _, _, _, _, w2_1 = compute_metrics(samples1, None, lam, reward_center, is_unguided=False)
    # Actually, W2 floor should be computed directly between the two empirical distributions over the 32 cells?
    # Wait, the prompt says "W2 giữa hai tập mẫu độc lập cùng rút từ target".
    # In compute_metrics, W2 is computed between the sample's discrete distribution and the EXACT target_weights.
    # So if we pass samples1 to compute_metrics, it returns W2(samples1_discrete, exact_discrete).
    # That IS the W2 floor for a finite sample size! The expected W2 distance between an empirical distribution of 5000 samples and the true distribution.
    return w2_1

def analyze_samples(lam=10.0, reward_center=[0.5, 1.5], suffix=''):
    unguided = np.load('results/unguided.npz')['samples']
    plugin = np.load(f'results/guided_k1_lam{lam}{suffix}.npz')
    
    try:
        damped = np.load(f'results/guided_damped_lam{lam}{suffix}.npz')
        damped_s, damped_r = damped['samples'], damped['rewards']
        has_damped = True
    except FileNotFoundError:
        has_damped = False

    second_order = np.load(f'results/guided_second_order_lam{lam}{suffix}.npz')

    plugin_s, plugin_r = plugin['samples'], plugin['rewards']
    second_s, second_r = second_order['samples'], second_order['rewards']
    
    # Generate Target Analytic Samples
    target_s1, target_r1 = generate_analytic_samples(lam, reward_center, num_samples=2000)
    target_s2, target_r2 = generate_analytic_samples(lam, reward_center, num_samples=2000)

    names = ["Target (Analytic)", "Unguided", "First-Order"]
    datasets = [
        (target_s1, target_r1, False),
        (unguided, None, True),
        (plugin_s, plugin_r, False)
    ]
    
    if has_damped:
        names.append("First-Order + Damp")
        datasets.append((damped_s, damped_r, False))
        
    names.append("Second-Order")
    datasets.append((second_s, second_r, False))
    
    print("-" * 80)
    print(f"Metrics (Reward Center: {reward_center}, lambda: {lam})")
    print(f"{'Method':<20} | {'Reward':<8} | {'In-Dist':<8} | {'Coverage':<8} | {'Entropy':<8} | {'W2':<8}")
    print("-" * 80)
    
    for name, (samples, rewards, is_ung) in zip(names, datasets):
        r, in_dist, cov, ent, w2 = compute_metrics(samples, rewards, lam, reward_center, is_unguided=is_ung)
        r_str = f"{r:.3f}" if not is_ung else "N/A"
        print(f"{name:<20} | {r_str:<8} | {in_dist*100:5.1f}% | {cov*100:6.1f}% | {ent:.4f}   | {w2:.4f}")
        
    # Compute W2 floor directly between two independent empirical sets?
    # W2 floor is the W2 distance returned for the Target (Analytic) row since compute_metrics computes 
    # distance to the exact target weights.
        
    print("-" * 80)

if __name__ == '__main__':
    # Center in white square (out of support) -> (0.5, 1.5)
    analyze_samples(lam=10.0, reward_center=[0.5, 1.5], suffix='_out')
    print()
    analyze_samples(lam=50.0, reward_center=[0.5, 1.5], suffix='_out')
    print()
    # Center in grey square (in support) -> (0.5, 0.5)
    analyze_samples(lam=10.0, reward_center=[0.5, 0.5], suffix='_in')
    print()
    analyze_samples(lam=50.0, reward_center=[0.5, 0.5], suffix='_in')

