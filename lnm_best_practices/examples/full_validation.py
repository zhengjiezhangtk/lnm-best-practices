"""
Full LNM Validation Example.

Demonstrates the complete LNM validation pipeline with null models:
1. Compute LNM
2. Spatial null model (location randomization)
3. Permutation test (symptom permutation with FWER correction)
4. Specificity testing (degree correlation)
5. Multiple comparison correction
"""

import numpy as np
import matplotlib.pyplot as plt

from lnm_best_practices.core.lnm import LNM
from lnm_best_practices.null_models.spatial import SpatialNullModel
from lnm_best_practices.null_models.permutation import PermutationTest
from lnm_best_practices.statistics.specificity import SpecificityTest
from lnm_best_practices.statistics.correction import fdr_correction


def main():
    np.random.seed(42)

    # ---------------------------------------------------------------
    # 1. Create synthetic data
    # ---------------------------------------------------------------
    n_subjects = 30
    n_parcels = 100

    # Lesion matrix
    M = np.zeros((n_subjects, n_parcels))
    for i in range(n_subjects):
        n_affected = np.random.randint(5, 15)
        affected = np.random.choice(n_parcels, n_affected, replace=False)
        M[i, affected] = 1.0

    # Connectome
    C = np.random.rand(n_parcels, n_parcels)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)

    # Symptoms (correlated with lesion location for realism)
    symptoms = np.random.randn(n_subjects)

    print("=" * 60)
    print("LNM Full Validation Pipeline")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 2. Compute empirical LNM
    # ---------------------------------------------------------------
    lnm = LNM(M, C)
    empirical_result = lnm.compute()
    empirical_map = empirical_result.lnm_map

    print(f"\n[1] Empirical LNM computed: shape={empirical_map.shape}")

    # ---------------------------------------------------------------
    # 3. Spatial null model
    # ---------------------------------------------------------------
    print("\n[2] Running spatial null model...")
    spatial_null = SpatialNullModel(random_state=42)
    spatial_null_maps = spatial_null.generate_null(M, n_permutations=200)
    spatial_p = spatial_null.compute_pvalue(empirical_map, spatial_null_maps)

    # FDR correction
    spatial_fdr_p, spatial_fdr_sig = fdr_correction(spatial_p, alpha=0.05)

    print(f"    Spatial null maps shape: {spatial_null_maps.shape}")
    print(f"    Significant parcels (FDR): {spatial_fdr_sig.sum()}")

    # ---------------------------------------------------------------
    # 4. Permutation test (symptom permutation + FWER)
    # ---------------------------------------------------------------
    print("\n[3] Running permutation test...")
    perm_test = PermutationTest(random_state=42)
    permuted_symptoms = perm_test.permute_symptoms(symptoms, n_permutations=200)

    # Generate null LNM maps from permuted symptoms
    null_maps = np.zeros((200, n_parcels))
    for i in range(200):
        sv_z = (permuted_symptoms[i] - permuted_symptoms[i].mean()) / permuted_symptoms[i].std()
        null_maps[i] = sv_z @ (M @ C)

    # FWER correction
    fwer_p, fwer_sig = perm_test.fwer_correction(empirical_map, null_maps, alpha=0.05)

    print(f"    Null maps shape: {null_maps.shape}")
    print(f"    Significant parcels (FWER): {fwer_sig.sum()}")

    # ---------------------------------------------------------------
    # 5. Specificity testing
    # ---------------------------------------------------------------
    print("\n[4] Running specificity tests...")
    degree_map = np.sum(C, axis=0)

    specificity = SpecificityTest(n_permutations=100)
    spec_result = specificity.run(
        empirical_map, degree_map,
        lesion_matrix=M, connectome=C, seed=42
    )

    print(f"    Degree correlation: r={spec_result.degree_correlation:.3f}, p={spec_result.degree_pvalue:.4f}")
    print(f"    Specificity index: {spec_result.specificity_index:.3f}")
    print(f"    Is specific (random lesion test): {spec_result.is_specific}")

    # ---------------------------------------------------------------
    # 6. Visualize results
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Empirical LNM
    axes[0, 0].bar(range(n_parcels), empirical_map)
    axes[0, 0].set_title("Empirical LNM Map")
    axes[0, 0].set_xlabel("Parcel")

    # Spatial null distribution (example parcel)
    axes[0, 1].hist(spatial_null_maps[:, 0], bins=30, alpha=0.7, label='Null')
    axes[0, 1].axvline(empirical_map[0], color='r', linestyle='--', label='Empirical')
    axes[0, 1].set_title("Spatial Null Distribution (Parcel 0)")
    axes[0, 1].legend()

    # Spatial p-values
    axes[0, 2].bar(range(n_parcels), spatial_p)
    axes[0, 2].axhline(0.05, color='r', linestyle='--', label='p=0.05')
    axes[0, 2].set_title("Spatial P-values")
    axes[0, 2].set_xlabel("Parcel")
    axes[0, 2].legend()

    # FWER p-values
    axes[1, 0].bar(range(n_parcels), fwer_p)
    axes[1, 0].axhline(0.05, color='r', linestyle='--', label='p=0.05')
    axes[1, 0].set_title("FWER-corrected P-values")
    axes[1, 0].set_xlabel("Parcel")
    axes[1, 0].legend()

    # LNM vs Degree
    axes[1, 1].scatter(degree_map, empirical_map, alpha=0.6)
    axes[1, 1].set_xlabel("Node Degree")
    axes[1, 1].set_ylabel("LNM Value")
    axes[1, 1].set_title(f"LNM vs Degree (r={spec_result.degree_correlation:.3f})")

    # Significant parcels
    sig_map = np.zeros(n_parcels)
    sig_map[spatial_fdr_sig] = 1
    sig_map[fwer_sig] = 2
    axes[1, 2].bar(range(n_parcels), sig_map)
    axes[1, 2].set_title("Significant Parcels")
    axes[1, 2].set_xlabel("Parcel")
    axes[1, 2].set_ylabel("0=NS, 1=FDR, 2=FWER")

    plt.tight_layout()
    plt.savefig("full_validation_example.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Spatial null: {spatial_fdr_sig.sum()} parcels significant (FDR)")
    print(f"  - Permutation:  {fwer_sig.sum()} parcels significant (FWER)")
    print(f"  - Specificity:  index={spec_result.specificity_index:.3f}")
    print(f"\nSaved plot to full_validation_example.png")


if __name__ == "__main__":
    main()
