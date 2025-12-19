"""
Performance benchmarks for parsing operations.

Run with: python -m benchmarks.bench_parsing
"""
import time
from pathlib import Path
from statistics import mean, stdev

from src.simulation.results import ResultsParser
from src.baseline.calibrator import BaselineCalibrator


def benchmark_results_parsing(iterations: int = 100):
    """Benchmark results CSV parsing."""
    output_dir = Path(__file__).parent.parent / "output_final"

    if not output_dir.exists():
        return {"test": "results_parsing", "status": "skipped", "reason": "No output directory"}

    parser = ResultsParser()
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        parser.parse(output_dir)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "test": "results_parsing",
        "iterations": iterations,
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
    }


def benchmark_parameter_extraction(iterations: int = 100):
    """Benchmark IDF parameter extraction."""
    idf_path = Path(__file__).parent.parent / "sjostaden_7zone.idf"

    if not idf_path.exists():
        return {"test": "param_extraction", "status": "skipped", "reason": "No IDF file"}

    idf_content = idf_path.read_text()
    calibrator = BaselineCalibrator.__new__(BaselineCalibrator)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        calibrator._extract_parameters(idf_content)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "test": "param_extraction",
        "iterations": iterations,
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
    }


def benchmark_idf_modification(iterations: int = 100):
    """Benchmark IDF parameter modification."""
    idf_path = Path(__file__).parent.parent / "sjostaden_7zone.idf"

    if not idf_path.exists():
        return {"test": "param_modification", "status": "skipped", "reason": "No IDF file"}

    idf_content = idf_path.read_text()
    calibrator = BaselineCalibrator.__new__(BaselineCalibrator)

    params = {
        "infiltration": 0.04,
        "heat_recovery": 0.80,
        "window_u": 0.9,
    }

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        calibrator._modify_idf(idf_content, params)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "test": "param_modification",
        "iterations": iterations,
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks():
    """Run all parsing benchmarks."""
    print("=" * 60)
    print("Parsing Performance Benchmarks")
    print("=" * 60)

    benchmarks = [
        ("Results CSV Parsing", benchmark_results_parsing),
        ("Parameter Extraction", benchmark_parameter_extraction),
        ("IDF Modification", benchmark_idf_modification),
    ]

    print(f"\n{'Benchmark':<25} {'Mean (ms)':<12} {'Stdev':<10} {'Status':<15}")
    print("-" * 60)

    for name, func in benchmarks:
        try:
            result = func()
            if result.get("status") == "skipped":
                print(f"{name:<25} {'--':<12} {'--':<10} {result['reason']:<15}")
            else:
                print(f"{name:<25} {result['mean_ms']:<12.3f} {result['stdev_ms']:<10.3f} {'OK':<15}")
        except Exception as e:
            print(f"{name:<25} {'--':<12} {'--':<10} ERROR: {e}")

    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
