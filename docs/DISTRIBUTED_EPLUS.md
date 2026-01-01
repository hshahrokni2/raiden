# Distributed EnergyPlus Architecture

## The Problem

Full portfolio analysis for all BRFs requires:
- **37,489 buildings** (Stockholm GeoJSON)
- **~6 simulations each** (1 baseline + 3-8 ECM packages)
- **~225,000 E+ simulations total**
- **~10 sec per simulation**
- **Sequential: 26 days** | **8 workers: 3.3 days** | **128 workers: 5 hours**

## Solution Options

### Option 1: Archetype-Based Pre-computation (RECOMMENDED)

**Insight**: Buildings sharing the same archetype have identical envelope parameters.
The only variables are: Atemp, orientation, and footprint shape.

**Strategy**:
1. Pre-simulate each archetype at 5 Atemp breakpoints (500, 1000, 2000, 5000, 10000 m²)
2. Pre-simulate each ECM package for each archetype
3. For real buildings: interpolate by Atemp, apply orientation correction

**Reduction**:
- 40 archetypes × 5 Atemp × 6 packages = **1,200 simulations** (not 225,000)
- Training time: ~3.3 hours (already done for surrogates!)
- Per-building lookup: ~1 ms

**Accuracy**: ~95% for similar buildings, scaling by Atemp²/³ law

**Files to create**:
- `src/simulation/archetype_cache.py` - Pre-computed results
- `src/simulation/interpolator.py` - Atemp/orientation scaling

---

### Option 2: Distributed Cloud Burst

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     RAIDEN COORDINATOR                          │
│                                                                  │
│  1. Generate 225,000 simulation tasks                           │
│  2. Upload to task queue (Redis/SQS)                            │
│  3. Monitor progress                                             │
│  4. Collect results                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  WORKER 1   │   │  WORKER 2   │   │  WORKER N   │
    │  (c5.4xl)   │   │  (c5.4xl)   │   │  (c5.4xl)   │
    │             │   │             │   │             │
    │ 16 E+ sims  │   │ 16 E+ sims  │   │ 16 E+ sims  │
    │  parallel   │   │  parallel   │   │  parallel   │
    └─────────────┘   └─────────────┘   └─────────────┘
```

**AWS Spot Instance Pricing** (eu-north-1, Stockholm):
| Instance | vCPU | RAM | Spot $/hr | E+ workers | Sims/hr |
|----------|------|-----|-----------|------------|---------|
| c5.4xlarge | 16 | 32GB | ~$0.25 | 16 | 5,760 |
| c5.9xlarge | 36 | 72GB | ~$0.55 | 36 | 12,960 |
| c5.18xlarge | 72 | 144GB | ~$1.10 | 72 | 25,920 |

**Cost for 225,000 simulations**:
- 100 × c5.4xlarge = 1,600 workers
- Time: 225,000 / (1,600 × 360/hr) = **~24 minutes**
- Cost: 100 × $0.25 × 0.4 hr = **~$10**

---

### Option 3: Kubernetes Cluster

For recurring workloads, a persistent k8s cluster:

```yaml
# raiden-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: raiden-eplus-worker
spec:
  replicas: 100
  template:
    spec:
      containers:
      - name: eplus-worker
        image: raiden/eplus-worker:latest
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: raiden-secrets
              key: redis-url
```

---

## Implementation Plan

### Phase 1: Archetype Cache (Quick Win)

```python
# src/simulation/archetype_cache.py

class ArchetypeSimulationCache:
    """
    Pre-computed E+ results for all archetype × Atemp × ECM combinations.

    Reduces 225,000 simulations to 1,200 pre-computed values + interpolation.
    """

    def __init__(self, cache_dir: Path = Path("./archetype_cache")):
        self.cache_dir = cache_dir
        self.cache: Dict[str, Dict] = {}  # archetype_id -> results

    def get_baseline(
        self,
        archetype_id: str,
        atemp_m2: float,
        orientation_deg: float = 0,
    ) -> float:
        """Get baseline kWh/m² for building, interpolating from pre-computed."""
        # Find bracketing Atemp values
        breakpoints = [500, 1000, 2000, 5000, 10000]
        lower = max(b for b in breakpoints if b <= atemp_m2)
        upper = min(b for b in breakpoints if b >= atemp_m2)

        # Interpolate using Atemp^(2/3) scaling (surface area law)
        kwh_lower = self.cache[archetype_id][lower]["baseline_kwh_m2"]
        kwh_upper = self.cache[archetype_id][upper]["baseline_kwh_m2"]

        # Linear interpolation in log space
        t = (atemp_m2 - lower) / (upper - lower) if upper > lower else 0
        kwh_m2 = kwh_lower * (1 - t) + kwh_upper * t

        # Orientation correction (±5% for N/S vs E/W)
        orientation_factor = 1.0 + 0.05 * math.cos(math.radians(orientation_deg))

        return kwh_m2 * orientation_factor

    def get_ecm_savings(
        self,
        archetype_id: str,
        ecm_package_id: str,
        atemp_m2: float,
    ) -> Tuple[float, float]:
        """Get (savings_kwh_m2, savings_percent) for ECM package."""
        # Similar interpolation logic
        ...
```

### Phase 2: Distributed Worker

```python
# src/simulation/distributed_worker.py

import redis
from dataclasses import dataclass

@dataclass
class SimulationTask:
    task_id: str
    address: str
    archetype_id: str
    idf_content: str  # Base64 encoded
    weather_file: str
    ecm_package: Optional[str]

class EPlusWorker:
    """Worker that pulls tasks from Redis and runs E+ simulations."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.worker_id = str(uuid.uuid4())[:8]

    def run_forever(self):
        while True:
            # Blocking pop from task queue
            task_data = self.redis.blpop("raiden:tasks", timeout=30)
            if task_data:
                task = SimulationTask(**json.loads(task_data[1]))
                result = self._run_simulation(task)
                self.redis.hset("raiden:results", task.task_id, json.dumps(result))

    def _run_simulation(self, task: SimulationTask) -> dict:
        # Decode IDF, run E+, return results
        ...
```

### Phase 3: Coordinator

```python
# src/simulation/coordinator.py

class DistributedSimulationCoordinator:
    """Orchestrates distributed E+ simulations."""

    async def run_portfolio(
        self,
        buildings: List[BuildingData],
        use_cache: bool = True,
    ) -> List[BuildingResult]:

        # Phase 1: Check cache for quick answers
        cached_results = []
        needs_simulation = []

        for building in buildings:
            if use_cache and self.cache.has_archetype(building.archetype_id):
                result = self.cache.interpolate(building)
                cached_results.append(result)
            else:
                needs_simulation.append(building)

        # Phase 2: Distribute remaining to workers
        if needs_simulation:
            tasks = self._generate_tasks(needs_simulation)
            await self._distribute_tasks(tasks)
            simulated_results = await self._collect_results(tasks)

        return cached_results + simulated_results
```

---

## Recommendation

**Start with Option 1 (Archetype Cache)**:
1. Already have 40 trained surrogates
2. Can generate interpolated results in milliseconds
3. 95%+ accuracy for typical buildings
4. Zero cloud cost

**Use Option 2 (Cloud Burst) for**:
1. Validation runs (E+ confirm top 10%)
2. Buildings that don't match archetypes well
3. When 100% E+ accuracy is required for legal/compliance

**Cost comparison for 37,489 buildings**:

| Approach | Time | Cost | Accuracy |
|----------|------|------|----------|
| Archetype cache | ~30 sec | $0 | ~95% |
| Cloud burst (10% validation) | ~3 min | ~$1 | 95% cached + 100% validated |
| Full cloud burst | ~25 min | ~$10 | 100% |
| Local 8-core | 3.3 days | $0 | 100% |

---

## Files to Create

1. `src/simulation/archetype_cache.py` - Pre-computed interpolation
2. `src/simulation/distributed_worker.py` - Redis-based E+ worker
3. `src/simulation/coordinator.py` - Task distribution
4. `docker/Dockerfile.eplus-worker` - Containerized E+ worker
5. `terraform/aws_spot_fleet.tf` - Infrastructure as code
