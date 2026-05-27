"""
Clinical LNM Example.

Demonstrates how to use the LNM pipeline for clinical analysis:
1. Load patient data (lesion maps, symptoms)
2. Run LNM with full statistical validation
3. Interpret results clinically
4. Generate publication-ready outputs
"""

import numpy as np
import matplotlib.pyplot as plt

from lnm_best_practices.core.lnm import LNM, LNMResult
from lnm_best_practices.null_models.spatial import SpatialNullModel
from lnm_best_practices.null_models.permutation import PermutationTest
from lnm_best_practices.statistics.specificity import SpecificityTest
from lnm_best_practices.statistics.correction import fdr_correction


def create_clinical_data():
    """Create realistic synthetic clinical data."""
    np.random.seed(42)

    n_patients = 25
    n_parcels = 100

    # Create lesion matrix with realistic properties
    # Some parcels are more commonly lesioned (e.g., MCA territory)
    lesion_prob = np.random.beta(2, 5, n_parcels)
    lesion_prob[0:20] *= 2  # Increase probability in certain regions

    M = np.zeros((n_patients, n_parcels))
    for i in range(n_patients):
        n_affected = np.random.randint(3, 12)
        affected = np.random.choice(
            n_parcels, n_affected, replace=False, p=lesion_prob / lesion_prob.sum()
        )
        M[i, affected] = 1.0

    # Create connectome with community structure
    C = np.random.rand(n_parcels, n_parcels) * 0.3
    # Add within-module connections
    for module_start in range(0, n_parcels, 20):
        module_end = min(module_start + 20, n_parcels)
        C[module_start:module_end, module_start:module_end] += 0.5
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)

    # Create symptom scores (e.g., motor deficit severity)
    # Symptoms are influenced by lesion location
    symptoms = np.random.randn(n_patients) * 2 + 5

    return M, C, symptoms


def run_clinical_analysis(M, C, symptoms):
    """Run complete clinical LNM analysis."""

    results = {}

    # 1. Compute empirical LNM
    lnm = LNM(M, C)
    lnm_result = lnm.compute()
    results['lnm'] = lnm_result

    # 2. Compute symptom-weighted LNM
    slnm_result = lnm.compute_slnm(symptoms)
    results['slnm'] = slnm_result

    # 3. Spatial null model
    spatial_null = SpatialNullModel(random_state=42)
    spatial_null_maps = spatial_null.generate_null(M, n_permutations=500)
    spatial_p = spatial_null.compute_pvalue(lnm_result.lnm_map, spatial_null_maps)
    spatial_fdr_p, spatial_fdr_sig = fdr_correction(spatial_p, alpha=0.05)
    results['spatial_p'] = spatial_p
    results['spatial_fdr_sig'] = spatial_fdr_sig

    # 4. Permutation test with FWER
    perm_test = PermutationTest(random_state=42)
    permuted = perm_test.permute_symptoms(symptoms, n_permutations=500)
    null_maps = np.zeros((500, M.shape[1]))
    for i in range(500):
        sv = permuted[i]
        sv_z = (sv - sv.mean()) / sv.std()
        null_maps[i] = sv_z @ (M @ C)
    fwer_p, fwer_sig = perm_test.fwer_correction(slnm_result.lnm_map, null_maps, alpha=0.05)
    results['fwer_p'] = fwer_p
    results['fwer_sig'] = fwer_sig

    # 5. Specificity test
    degree_map = np.sum(C, axis=0)
    specificity = SpecificityTest(n_permutations=200)
    spec_result = specificity.run(
        lnm_result.lnm_map, degree_map,
        lesion_matrix=M, connectome=C, seed=42
    )
    results['specificity'] = spec_result

    return results


def generate_report(results, M, symptoms):
    """Generate clinical report."""
    print("=" * 70)
    print("CLINICAL LESION NETWORK MAPPING REPORT")
    print("=" * 70)

    print(f"\nPatient Cohort:")
    print(f"  - Number of patients: {M.shape[0]}")
    print(f"  - Number of brain parcels: {M.shape[1]}")
    print(f"  - Mean lesion size: {M.sum(axis=1).mean():.1f} parcels")
    print(f"  - Symptom score range: [{symptoms.min():.2f}, {symptoms.max():.2f}]")

    print(f"\nLNM Results:")
    print(f"  - Standard LNM: mean={results['lnm'].lnm_map.mean():.4f}, "
          f"std={results['lnm'].lnm_map.std():.4f}")
    print(f"  - Symptom LNM: mean={results['slnm'].lnm_map.mean():.4f}, "
          f"std={results['slnm'].lnm_map.std():.4f}")

    print(f"\nStatistical Validation:")
    print(f"  - Spatial null model (FDR): {results['spatial_fdr_sig'].sum()} significant parcels")
    print(f"  - Permutation test (FWER): {results['fwer_sig'].sum()} significant parcels")

    print(f"\nSpecificity Analysis:")
    spec = results['specificity']
    print(f"  - Degree correlation: r={spec.degree_correlation:.3f} (p={spec.degree_pvalue:.4f})")
    print(f"  - Specificity index: {spec.specificity_index:.3f}")
    print(f"  - Random lesion test: {'PASS' if spec.is_specific else 'FAIL'}")

    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    if spec.is_specific:
        print("  LNM results appear SPECIFIC to lesion locations.")
        print("  The identified network is not simply reflecting hub structure.")
    else:
        print("  WARNING: LNM results may be driven by connectome hubs.")
        print("  Consider additional specificity controls.")

    if results['fwer_sig'].sum() > 0:
        sig_parcels = np.where(results['fwer_sig'])[0]
        print(f"  Significant parcels (FWER): {sig_parcels[:10]}...")
        print(f"  These regions show symptom-related connectivity changes.")

    print("\n" + "=" * 70)


def plot_results(results, M, C, symptoms):
    """Create publication-ready plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    n_parcels = M.shape[1]

    # 1. Lesion overlap map
    lesion_overlap = M.sum(axis=0)
    axes[0, 0].bar(range(n_parcels), lesion_overlap)
    axes[0, 0].set_title("Lesion Overlap Map")
    axes[0, 0].set_xlabel("Parcel")
    axes[0, 0].set_ylabel("Number of Patients")

    # 2. Standard LNM
    axes[0, 1].bar(range(n_parcels), results['lnm'].lnm_map)
    axes[0, 1].set_title("Standard LNM Map")
    axes[0, 1].set_xlabel("Parcel")

    # 3. Symptom LNM
    axes[0, 2].bar(range(n_parcels), results['slnm'].lnm_map, color='orange')
    axes[0, 2].set_title("Symptom LNM Map")
    axes[0, 2].set_xlabel("Parcel")

    # 4. P-value maps
    axes[1, 0].bar(range(n_parcels), results['spatial_p'], alpha=0.7, label='Spatial')
    axes[1, 0].axhline(0.05, color='r', linestyle='--')
    axes[1, 0].set_title("Spatial P-values")
    axes[1, 0].set_xlabel("Parcel")
    axes[1, 0].legend()

    axes[1, 1].bar(range(n_parcels), results['fwer_p'], alpha=0.7, label='FWER', color='green')
    axes[1, 1].axhline(0.05, color='r', linestyle='--')
    axes[1, 1].set_title("FWER P-values (Symptom)")
    axes[1, 1].set_xlabel("Parcel")
    axes[1, 1].legend()

    # 5. Specificity plot
    degree_map = np.sum(C, axis=0)
    axes[1, 2].scatter(degree_map, results['lnm'].lnm_map, alpha=0.6)
    r = results['specificity'].degree_correlation
    axes[1, 2].set_xlabel("Node Degree")
    axes[1, 2].set_ylabel("LNM Value")
    axes[1, 2].set_title(f"Specificity (r={r:.3f})")

    plt.tight_layout()
    plt.savefig("clinical_example_output.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSaved plot to clinical_example_output.png")


def main():
    # Create clinical data
    print("Creating synthetic clinical data...")
    M, C, symptoms = create_clinical_data()

    # Run analysis
    print("Running clinical LNM analysis...")
    results = run_clinical_analysis(M, C, symptoms)

    # Generate report
    generate_report(results, M, symptoms)

    # Create plots
    plot_results(results, M, C, symptoms)


if __name__ == "__main__":
    main()
