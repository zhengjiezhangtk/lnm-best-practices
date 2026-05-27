"""
Basic Lesion Network Mapping Example.

Demonstrates the core LNM pipeline:
1. Create lesion matrix and connectome
2. Compute standard LNM
3. Compute symptom-based LNM (sLNM)
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt

from lnm_best_practices.core.lnm import LNM


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # ---------------------------------------------------------------
    # 1. Create synthetic data
    # ---------------------------------------------------------------
    n_subjects = 20
    n_parcels = 100

    # Random lesion matrix (binary, normalized)
    M = np.zeros((n_subjects, n_parcels))
    for i in range(n_subjects):
        n_affected = np.random.randint(5, 15)
        affected = np.random.choice(n_parcels, n_affected, replace=False)
        M[i, affected] = 1.0

    # Random symmetric connectome
    C = np.random.rand(n_parcels, n_parcels)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)

    # Random symptom scores
    symptoms = np.random.randn(n_subjects)

    print(f"Lesion matrix shape: {M.shape}")
    print(f"Connectome shape: {C.shape}")
    print(f"Symptoms shape: {symptoms.shape}")

    # ---------------------------------------------------------------
    # 2. Compute standard LNM
    # ---------------------------------------------------------------
    lnm = LNM(M, C)
    result = lnm.compute()

    print(f"\nStandard LNM:")
    print(f"  Map shape: {result.lnm_map.shape}")
    print(f"  Mean: {result.lnm_map.mean():.4f}")
    print(f"  Std: {result.lnm_map.std():.4f}")

    # ---------------------------------------------------------------
    # 3. Compute symptom-based LNM (sLNM)
    # ---------------------------------------------------------------
    slnm_result = lnm.compute_slnm(symptoms)

    print(f"\nSymptom LNM:")
    print(f"  Map shape: {slnm_result.lnm_map.shape}")
    print(f"  Mean: {slnm_result.lnm_map.mean():.4f}")
    print(f"  Std: {slnm_result.lnm_map.std():.4f}")

    # ---------------------------------------------------------------
    # 4. Normalize maps
    # ---------------------------------------------------------------
    lnm_z = LNM.normalize_map(result.lnm_map, method='zscore')
    slnm_z = LNM.normalize_map(slnm_result.lnm_map, method='zscore')

    print(f"\nNormalized LNM (z-score):")
    print(f"  Mean: {lnm_z.mean():.6f}")
    print(f"  Std: {lnm_z.std():.6f}")

    # ---------------------------------------------------------------
    # 5. Visualize
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot LNM map
    axes[0].bar(range(n_parcels), result.lnm_map)
    axes[0].set_xlabel("Parcel")
    axes[0].set_ylabel("LNM Value")
    axes[0].set_title("Standard LNM Map")

    # Plot sLNM map
    axes[1].bar(range(n_parcels), slnm_result.lnm_map, color='orange')
    axes[1].set_xlabel("Parcel")
    axes[1].set_ylabel("sLNM Value")
    axes[1].set_title("Symptom-based LNM Map")

    # Plot correlation between LNM and sLNM
    axes[2].scatter(result.lnm_map, slnm_result.lnm_map, alpha=0.6)
    r = np.corrcoef(result.lnm_map, slnm_result.lnm_map)[0, 1]
    axes[2].set_xlabel("LNM")
    axes[2].set_ylabel("sLNM")
    axes[2].set_title(f"LNM vs sLNM (r={r:.3f})")

    plt.tight_layout()
    plt.savefig("basic_lnm_example.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nDone! Saved plot to basic_lnm_example.png")


if __name__ == "__main__":
    main()
