"""
Combination Generator - Generate valid ECM combinations.

Takes valid ECMs and generates all reasonable combinations:
- Single ECMs
- Pairs
- Packages (3-4 ECMs)
- Full retrofit

Prunes dominated and incompatible combinations.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from itertools import combinations

from .catalog import ECM, ECMParameter


@dataclass
class ECMVariant:
    """A specific ECM with chosen parameter values."""
    ecm: ECM
    parameters: Dict[str, any]  # Parameter name -> value

    @property
    def id(self) -> str:
        """Unique identifier for this variant."""
        param_str = "_".join(f"{k}{v}" for k, v in sorted(self.parameters.items()))
        return f"{self.ecm.id}_{param_str}" if param_str else self.ecm.id


@dataclass
class ECMCombination:
    """A combination of ECM variants to simulate."""
    variants: List[ECMVariant]
    name: str

    @property
    def id(self) -> str:
        """Unique identifier for this combination."""
        return "+".join(v.id for v in self.variants)

    @property
    def ecm_ids(self) -> Set[str]:
        """Set of ECM IDs in this combination."""
        return {v.ecm.id for v in self.variants}


# Incompatible ECM pairs (can't do both)
INCOMPATIBLE_PAIRS = {
    ("wall_external_insulation", "wall_internal_insulation"),  # Pick one
    ("ftx_upgrade", "ftx_installation"),  # Either upgrade or new install
}

# ECMs that should be done together
SYNERGISTIC_PAIRS = {
    ("air_sealing", "ftx_upgrade"),  # Seal before upgrading ventilation
    ("roof_insulation", "solar_pv"),  # Do roof work together
}


class CombinationGenerator:
    """
    Generate ECM combinations for simulation.

    Usage:
        generator = CombinationGenerator()
        combinations = generator.generate(
            valid_ecms=[...],
            max_combination_size=4
        )
    """

    def __init__(
        self,
        incompatible_pairs: Set[Tuple[str, str]] = None,
        synergistic_pairs: Set[Tuple[str, str]] = None
    ):
        self.incompatible = incompatible_pairs or INCOMPATIBLE_PAIRS
        self.synergistic = synergistic_pairs or SYNERGISTIC_PAIRS

    def generate_variants(self, ecm: ECM) -> List[ECMVariant]:
        """
        Generate all parameter variants for an ECM.

        For ECM with parameters [a: [1,2], b: [x,y]], generates:
        - {a: 1, b: x}
        - {a: 1, b: y}
        - {a: 2, b: x}
        - {a: 2, b: y}
        """
        if not ecm.parameters:
            return [ECMVariant(ecm=ecm, parameters={})]

        # Generate all parameter combinations
        param_names = [p.name for p in ecm.parameters]
        param_values = [p.values for p in ecm.parameters]

        from itertools import product
        variants = []
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            variants.append(ECMVariant(ecm=ecm, parameters=params))

        return variants

    def are_compatible(self, ecm_ids: Set[str]) -> bool:
        """Check if ECMs are compatible (no incompatible pairs)."""
        for pair in self.incompatible:
            if pair[0] in ecm_ids and pair[1] in ecm_ids:
                return False
        return True

    def generate(
        self,
        valid_ecms: List[ECM],
        max_combination_size: int = 4,
        include_baseline: bool = True
    ) -> List[ECMCombination]:
        """
        Generate all valid ECM combinations.

        Args:
            valid_ecms: List of ECMs that passed constraint checking
            max_combination_size: Maximum ECMs per combination
            include_baseline: Include empty combination (baseline)

        Returns:
            List of ECMCombination objects to simulate
        """
        result = []

        # Baseline (no ECMs)
        if include_baseline:
            result.append(ECMCombination(variants=[], name="Baseline"))

        # Generate variants for each ECM
        all_variants: Dict[str, List[ECMVariant]] = {}
        for ecm in valid_ecms:
            all_variants[ecm.id] = self.generate_variants(ecm)

        # Single ECMs (all variants)
        for ecm_id, variants in all_variants.items():
            for variant in variants:
                result.append(ECMCombination(
                    variants=[variant],
                    name=f"{variant.ecm.name}"
                ))

        # Combinations of 2 to max_size
        ecm_ids = list(all_variants.keys())
        for size in range(2, min(max_combination_size + 1, len(ecm_ids) + 1)):
            for combo_ids in combinations(ecm_ids, size):
                # Check compatibility
                if not self.are_compatible(set(combo_ids)):
                    continue

                # For combinations, use "middle" parameter values to limit explosion
                variants = []
                for ecm_id in combo_ids:
                    ecm_variants = all_variants[ecm_id]
                    # Pick middle variant (reasonable default)
                    middle_idx = len(ecm_variants) // 2
                    variants.append(ecm_variants[middle_idx])

                name = " + ".join(v.ecm.name for v in variants)
                result.append(ECMCombination(variants=variants, name=name))

        return result

    def generate_packages(
        self,
        valid_ecms: List[ECM]
    ) -> Dict[str, ECMCombination]:
        """
        Generate predefined ECM packages.

        Returns:
            Dict with 'basic', 'standard', 'premium' packages
        """
        packages = {}

        # Basic: Quick wins (low cost, low disruption)
        basic_ids = {'led_lighting', 'air_sealing', 'smart_thermostats'}
        basic_variants = []
        for ecm in valid_ecms:
            if ecm.id in basic_ids:
                basic_variants.append(ECMVariant(ecm=ecm, parameters={}))
        if basic_variants:
            packages['basic'] = ECMCombination(
                variants=basic_variants,
                name="Basic Package (Quick Wins)"
            )

        # Standard: Moderate investment
        standard_ids = basic_ids | {'ftx_upgrade', 'roof_insulation'}
        standard_variants = []
        for ecm in valid_ecms:
            if ecm.id in standard_ids:
                standard_variants.append(ECMVariant(ecm=ecm, parameters={}))
        if standard_variants:
            packages['standard'] = ECMCombination(
                variants=standard_variants,
                name="Standard Package"
            )

        # Premium: Deep retrofit
        premium_ids = standard_ids | {'window_replacement', 'solar_pv'}
        premium_variants = []
        for ecm in valid_ecms:
            if ecm.id in premium_ids:
                premium_variants.append(ECMVariant(ecm=ecm, parameters={}))
        if premium_variants:
            packages['premium'] = ECMCombination(
                variants=premium_variants,
                name="Premium Package (Deep Retrofit)"
            )

        return packages

    def estimate_combination_count(
        self,
        valid_ecms: List[ECM],
        max_size: int = 4
    ) -> int:
        """Estimate number of combinations (before generation)."""
        n = len(valid_ecms)
        count = 1  # Baseline
        for size in range(1, min(max_size + 1, n + 1)):
            # Approximate: C(n, size) * average variants
            from math import comb
            avg_variants = 3  # Rough estimate
            count += comb(n, size) * (avg_variants if size == 1 else 1)
        return count
