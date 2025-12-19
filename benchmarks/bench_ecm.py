"""
Performance benchmarks for ECM modifications.

Run with: python -m benchmarks.bench_ecm
"""
import time
from pathlib import Path
from statistics import mean, stdev

from src.ecm.idf_modifier import IDFModifier


def load_idf():
    """Load the Sjostaden IDF for benchmarking."""
    idf_path = Path(__file__).parent.parent / "sjostaden_7zone.idf"
    if not idf_path.exists():
        raise FileNotFoundError(f"Benchmark IDF not found: {idf_path}")
    return idf_path.read_text()


def benchmark_ecm(ecm_id: str, params: dict, iterations: int = 100):
    """Benchmark a single ECM application."""
    idf_content = load_idf()
    modifier = IDFModifier()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        modifier._apply_ecm(idf_content, ecm_id, params)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "ecm": ecm_id,
        "iterations": iterations,
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_chained_ecms(iterations: int = 50):
    """Benchmark applying multiple ECMs in sequence."""
    idf_content = load_idf()
    modifier = IDFModifier()

    ecms = [
        ("air_sealing", {"reduction_factor": 0.5}),
        ("window_replacement", {"u_value": 0.8, "shgc": 0.5}),
        ("led_lighting", {"watts_per_m2": 6.0}),
    ]

    times = []
    for _ in range(iterations):
        content = idf_content
        start = time.perf_counter()
        for ecm_id, params in ecms:
            content = modifier._apply_ecm(content, ecm_id, params)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "test": "chained_3_ecms",
        "iterations": iterations,
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Raiden Performance Benchmarks")
    print("=" * 60)

    # IDF info
    idf_content = load_idf()
    print(f"\nIDF size: {len(idf_content):,} bytes")
    print(f"IDF lines: {idf_content.count(chr(10)):,}")

    # Individual ECM benchmarks
    ecms = [
        ("air_sealing", {"reduction_factor": 0.5}),
        ("window_replacement", {"u_value": 0.8, "shgc": 0.5}),
        ("led_lighting", {"watts_per_m2": 6.0}),
        ("smart_thermostats", {"setback_deg": 3.0}),
    ]

    print("\n" + "-" * 60)
    print("Individual ECM Performance (100 iterations)")
    print("-" * 60)
    print(f"{'ECM':<25} {'Mean (ms)':<12} {'Stdev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)

    for ecm_id, params in ecms:
        try:
            result = benchmark_ecm(ecm_id, params)
            print(f"{result['ecm']:<25} {result['mean_ms']:<12.3f} {result['stdev_ms']:<10.3f} {result['min_ms']:<10.3f} {result['max_ms']:<10.3f}")
        except Exception as e:
            print(f"{ecm_id:<25} ERROR: {e}")

    # Chained ECM benchmark
    print("\n" + "-" * 60)
    print("Chained ECM Performance (50 iterations)")
    print("-" * 60)

    try:
        result = benchmark_chained_ecms()
        print(f"3 ECMs chained: {result['mean_ms']:.3f} ms (stdev: {result['stdev_ms']:.3f})")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_benchmarks()
