"""Stage 1: Trust Layer components."""

from .wrapper import (
    TrustWrapper,
    TrustVector,
    TrustPrediction,
    TemperatureScaling,
    OODDetector,
    StabilityChecker,
    SymmetryChecker,
    selective_prediction_eval
)

from .metrics import (
    SelectiveMetrics,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_ece,
    compute_brier_score,
    selective_accuracy_at_coverage,
    coverage_accuracy_curve,
    compute_auc_coverage_accuracy,
    aggregate_trust_scores
)

from .threshold_search import (
    ThresholdConfig,
    OptimizationResult,
    optimize_thresholds,
    grid_search_weights,
    coverage_constrained_search,
    pareto_frontier_search,
    compare_learned_vs_fixed
)

from .dscript_wrapper import (
    DScriptTrustWrapper,
    DScriptStabilityChecker,
    DScriptSymmetryChecker,
    selective_prediction_dscript
)

from .interface_confidence import (
    InterfaceConfidence,
    InterfaceMetrics,
    extract_dscript_contact_map,
    compute_interface_trust_dscript
)

from .deformation_stability import (
    DeformationStability,
    DeformationMetrics,
    SequenceDeformationStability,
    EmbeddingDeformationStability,
    compute_structure_deformation_trust,
    compute_sequence_deformation_trust,
    compute_embedding_deformation_trust
)

__all__ = [
    # Wrapper
    'TrustWrapper',
    'TrustVector',
    'TrustPrediction',
    'TemperatureScaling',
    'OODDetector',
    'StabilityChecker',
    'SymmetryChecker',
    'selective_prediction_eval',
    # Metrics
    'SelectiveMetrics',
    'compute_accuracy',
    'compute_precision_recall_f1',
    'compute_ece',
    'compute_brier_score',
    'selective_accuracy_at_coverage',
    'coverage_accuracy_curve',
    'compute_auc_coverage_accuracy',
    'aggregate_trust_scores',
    # Threshold search
    'ThresholdConfig',
    'OptimizationResult',
    'optimize_thresholds',
    'grid_search_weights',
    'coverage_constrained_search',
    'pareto_frontier_search',
    'compare_learned_vs_fixed',
    # D-SCRIPT wrapper
    'DScriptTrustWrapper',
    'DScriptStabilityChecker',
    'DScriptSymmetryChecker',
    'selective_prediction_dscript',
    # Interface confidence (protein-specific)
    'InterfaceConfidence',
    'InterfaceMetrics',
    'extract_dscript_contact_map',
    'compute_interface_trust_dscript',
    # Deformation stability (protein-specific)
    'DeformationStability',
    'DeformationMetrics',
    'SequenceDeformationStability',
    'EmbeddingDeformationStability',
    'compute_structure_deformation_trust',
    'compute_sequence_deformation_trust',
    'compute_embedding_deformation_trust'
]
