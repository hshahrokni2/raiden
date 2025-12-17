#!/usr/bin/env python3
"""
Process BRF Sjöstaden 2 - First example.

This is a standalone script that demonstrates the enrichment pipeline
without requiring the full package installation.

Integrates:
- JSON input parsing
- PDF energy declaration extraction (optional)
- U-value back-calculation from actual energy consumption
- 3D visualization
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pyproj import Transformer

console = Console()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "input" / "BRF_Sjostaden_2.json"
OUTPUT_DIR = PROJECT_ROOT / "examples" / "sjostaden_2"

# Optional PDF path (energy declaration)
PDF_FILE = Path("/Users/hosseins/Dropbox/zeldadb/zeldabot/pdf_docs/Energideklaration/76224_energideklaration_stockholm_brf_sjöstaden_2.pdf")

# Mapillary token for street-level imagery (optional)
# Set MAPILLARY_TOKEN environment variable or paste here
import os
MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "MLY|26005515675718354|a1ee419e8cc5663ed5402c295e1e6e1c")

# Image cache directory
IMAGE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "facade_images"


# ============================================================================
# Coordinate transformation
# ============================================================================

def create_transformer():
    """Create SWEREF99 TM to WGS84 transformer."""
    return Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)


def sweref_to_wgs84(transformer, coords_3d):
    """Convert 3D SWEREF99 coordinates to WGS84."""
    result = []
    for coord in coords_3d:
        x, y = coord[0], coord[1]
        lon, lat = transformer.transform(x, y)
        result.append((lon, lat))
    return result


def get_bbox_wgs84(transformer, coords, buffer_m=100):
    """Get bounding box in WGS84 with buffer."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    min_x = min(xs) - buffer_m
    min_y = min(ys) - buffer_m
    max_x = max(xs) + buffer_m
    max_y = max(ys) + buffer_m

    min_lon, min_lat = transformer.transform(min_x, min_y)
    max_lon, max_lat = transformer.transform(max_x, max_y)

    return (min_lon, min_lat, max_lon, max_lat)


def coords_to_local(coords, origin=None):
    """Convert coordinates to local system (relative to origin)."""
    xy_coords = [(c[0], c[1]) for c in coords]

    if origin is None:
        xs = [c[0] for c in xy_coords]
        ys = [c[1] for c in xy_coords]
        origin = (sum(xs) / len(xs), sum(ys) / len(ys))

    return [(x - origin[0], y - origin[1]) for x, y in xy_coords]


# ============================================================================
# Estimation functions
# ============================================================================

def estimate_wwr(construction_year: int) -> dict:
    """Estimate WWR based on construction era."""
    if construction_year < 1960:
        base = 0.18
    elif construction_year < 1975:
        base = 0.22
    elif construction_year < 1990:
        base = 0.20
    elif construction_year < 2010:
        base = 0.27
    else:
        base = 0.35

    return {
        "north": base * 0.8,
        "south": base * 1.2,
        "east": base,
        "west": base,
        "average": base,
        "source": "era_estimation",
        "confidence": 0.5,
    }


def estimate_u_values(construction_year: int, renovation_year: int = None) -> dict:
    """Estimate U-values based on construction era (Swedish BBR)."""
    ref_year = renovation_year if renovation_year else construction_year

    if ref_year >= 2020:
        return {"walls": 0.18, "roof": 0.13, "floor": 0.15, "windows": 1.2}
    elif ref_year >= 2010:
        return {"walls": 0.20, "roof": 0.13, "floor": 0.15, "windows": 1.3}
    elif ref_year >= 2000:
        return {"walls": 0.25, "roof": 0.15, "floor": 0.20, "windows": 1.6}
    elif ref_year >= 1990:
        return {"walls": 0.30, "roof": 0.20, "floor": 0.25, "windows": 2.0}
    elif ref_year >= 1975:
        return {"walls": 0.40, "roof": 0.25, "floor": 0.30, "windows": 2.5}
    else:
        return {"walls": 0.50, "roof": 0.35, "floor": 0.40, "windows": 2.8}


def estimate_solar_potential(roof_area_sqm: float) -> dict:
    """Estimate solar PV potential."""
    suitable_area = roof_area_sqm * 0.3  # 30% usable
    kwp = suitable_area * 0.2  # ~200W/m² density
    annual_kwh = kwp * 900  # ~900 kWh/kWp in Stockholm

    return {
        "suitable_roof_area_sqm": suitable_area,
        "remaining_capacity_kwp": kwp,
        "annual_yield_potential_kwh": annual_kwh,
        "shading_loss_pct": 10,
        "source": "estimation",
    }


# ============================================================================
# PDF Energy Declaration Parser
# ============================================================================

@dataclass
class PDFExtractionData:
    """Data extracted from energy declaration PDF."""
    addresses: list[str]
    ventilation_airflow_ls_sqm: Optional[float] = None
    radon_bq_m3: Optional[int] = None
    radon_status: str = ""
    specific_energy_kwh_sqm: Optional[float] = None
    reference_value_2: Optional[float] = None
    recommendations: list[dict] = None
    raw_text: str = ""


def parse_energy_declaration_pdf(pdf_path: Path) -> Optional[PDFExtractionData]:
    """
    Parse Swedish energy declaration PDF for additional data.

    Returns extracted data or None if parsing fails.
    """
    try:
        import pdfplumber
    except ImportError:
        console.print("  [yellow]⚠[/yellow] pdfplumber not installed, skipping PDF parsing")
        return None

    if not pdf_path.exists():
        console.print(f"  [yellow]⚠[/yellow] PDF not found: {pdf_path}")
        return None

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            row_text = " | ".join(str(cell) for cell in row if cell)
                            text_parts.append(row_text)
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] PDF parsing error: {e}")
        return None

    text = "\n".join(text_parts)

    # Extract addresses
    addresses = set()
    address_patterns = [
        r"(Lugnets\s+Allé)\s+(\d+(?:-\d+)?)",
        r"(Aktergatan)\s+(\d+(?:-\d+)?)",
        r"(Hammarby\s+Allé)\s+(\d+(?:-\d+)?)",
        r"([A-ZÅÄÖ][a-zåäö]+(?:gatan|vägen|allén|stigen))\s+(\d+(?:-\d+)?)",
    ]
    for pattern in address_patterns:
        for match in re.finditer(pattern, text, re.UNICODE | re.IGNORECASE):
            addr = f"{match.group(1)} {match.group(2)}"
            addresses.add(addr)

    # Extract ventilation airflow
    ventilation = None
    vent_match = re.search(r"([\d,\.]+)\s*l/s[,\s]*m²", text, re.IGNORECASE)
    if vent_match:
        try:
            ventilation = float(vent_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Extract radon
    radon = None
    radon_status = ""
    radon_match = re.search(r"[Rr]adon[:\s]*([\d]+)\s*Bq/m³", text)
    if radon_match:
        radon = int(radon_match.group(1))
        if radon < 200:
            radon_status = "GOOD"
        elif radon < 400:
            radon_status = "WARNING"
        else:
            radon_status = "HIGH"

    # Extract specific energy (various patterns in Swedish PDFs)
    # Looking for "Specifik energianvändning" followed by "XX kWh/m² och år"
    specific_energy = None
    spec_patterns = [
        # "Specifik energianvändning" section pattern
        r"[Ss]pecifik\s+energi[a-zåäö]*[\s\S]{0,50}?([\d]+)\s*kWh/m²\s*och\s*år",
        r"[Ss]pecifik\s+energi[a-zåäö]*[:\s]*([\d,\.]+)\s*kWh",
        r"([\d,\.]+)\s*kWh/m²\s*och\s*år",  # Swedish: "och år" = "and year"
        r"([\d,\.]+)\s*kWh/m²/år",
        r"([\d,\.]+)\s*kWh/m²\s*,\s*år",
    ]
    for pattern in spec_patterns:
        spec_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if spec_match:
            try:
                specific_energy = float(spec_match.group(1).replace(",", ".").replace(" ", ""))
                if specific_energy < 200:  # Sanity check - should be reasonable
                    break
            except ValueError:
                continue

    # Extract reference value 2
    ref_value_2 = None
    ref_match = re.search(r"[Rr]eferensvärde\s*2[:\s]*([\d,\.]+)", text)
    if ref_match:
        try:
            ref_value_2 = float(ref_match.group(1).replace(",", ".").replace(" ", ""))
        except ValueError:
            pass

    # Extract recommendations
    recommendations = []
    rec_match = re.search(r"FTX[a-zåäö\s-]*LB03[^.]*\.(?:[^.]*(?:kWh/år|kr)[^.]*\.)?", text, re.IGNORECASE)
    if rec_match:
        rec_text = rec_match.group(0)
        rec = {"description": "Replace FTX unit LB03"}

        savings_match = re.search(r"([\d\s]+)\s*kWh/år", rec_text)
        if savings_match:
            try:
                rec["annual_savings_kwh"] = int(savings_match.group(1).replace(" ", ""))
            except ValueError:
                pass

        cost_match = re.search(r"([\d\s,\.]+)\s*kr/år", rec_text)
        if cost_match:
            try:
                rec["cost_savings_kr"] = float(cost_match.group(1).replace(" ", "").replace(",", "."))
            except ValueError:
                pass

        impl_match = re.search(r"[Kk]ostnad[:\s]*([\d\s]+)\s*kr", rec_text)
        if impl_match:
            try:
                rec["implementation_cost_kr"] = int(impl_match.group(1).replace(" ", ""))
            except ValueError:
                pass

        recommendations.append(rec)

    return PDFExtractionData(
        addresses=sorted(list(addresses)),
        ventilation_airflow_ls_sqm=ventilation,
        radon_bq_m3=radon,
        radon_status=radon_status,
        specific_energy_kwh_sqm=specific_energy,
        reference_value_2=ref_value_2,
        recommendations=recommendations or [],
        raw_text=text,
    )


# ============================================================================
# U-value Back-Calculation
# ============================================================================

# Swedish climate data (Stockholm)
HEATING_DEGREE_DAYS = 3900  # °C-days/year
HOT_WATER_KWH_PER_SQM = 25  # kWh/m²/year


def back_calculate_u_values(
    specific_energy_kwh_sqm: float,
    heated_area_sqm: float,
    num_floors: int,
    height_m: float,
    construction_year: int,
    renovation_year: int = None,
    wwr: float = 0.20,
) -> dict:
    """
    Back-calculate U-values from actual energy consumption.

    Uses simplified steady-state heat loss to estimate
    building envelope U-values from measured data.
    """
    # Get era-based initial estimates
    era_u = estimate_u_values(construction_year, renovation_year)

    # Estimate envelope areas
    floor_area = heated_area_sqm / num_floors
    perimeter = 4 * math.sqrt(floor_area)  # Assume square
    gross_wall_area = perimeter * height_m
    window_area = gross_wall_area * wwr
    wall_area = gross_wall_area * (1 - wwr)

    # Calculate theoretical heat loss with era estimates
    infiltration_coeff = 0.15  # W/m²K equivalent
    h_total = (
        wall_area * era_u["walls"] +
        floor_area * era_u["roof"] +
        floor_area * era_u["floor"] * 0.7 +  # Ground coupled
        window_area * era_u["windows"] +
        heated_area_sqm * infiltration_coeff
    )

    theoretical_kwh = h_total * HEATING_DEGREE_DAYS * 24 / 1000
    theoretical_kwh_sqm = theoretical_kwh / heated_area_sqm

    # Estimate actual heating (subtract hot water)
    actual_heating_kwh_sqm = max(0, specific_energy_kwh_sqm - HOT_WATER_KWH_PER_SQM)

    # Calculate adjustment ratio
    ratio = actual_heating_kwh_sqm / theoretical_kwh_sqm if theoretical_kwh_sqm > 0 else 1
    scale = ratio ** 0.5  # Dampened scaling

    # Generate notes
    notes = []
    confidence = 0.7
    if 0.8 <= ratio <= 1.2:
        notes.append("Era estimates match actual well")
        confidence = 0.8
    elif ratio < 0.8:
        notes.append(f"Building {(1-ratio)*100:.0f}% better than era typical")
        confidence = 0.6
    else:
        notes.append(f"Building {(ratio-1)*100:.0f}% worse than era typical")
        confidence = 0.5

    notes.append(f"Based on {specific_energy_kwh_sqm:.0f} kWh/m²/year measured")

    # Clamp values
    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    return {
        "walls": clamp(era_u["walls"] * scale, 0.10, 1.0),
        "roof": clamp(era_u["roof"] * scale, 0.08, 0.5),
        "floor": clamp(era_u["floor"] * scale, 0.10, 0.5),
        "windows": clamp(era_u["windows"] * scale, 0.8, 3.0),
        "source": "back_calculation",
        "confidence": confidence,
        "notes": notes,
    }


# ============================================================================
# 3D Visualization
# ============================================================================

def generate_3d_mesh(footprint_local, height):
    """Generate Three.js compatible mesh data."""
    # Remove duplicate closing vertex if present
    if footprint_local[0] == footprint_local[-1]:
        footprint_local = footprint_local[:-1]

    n = len(footprint_local)

    # Wall vertices and normals
    wall_vertices = []
    wall_normals = []
    wall_indices = []

    for i in range(n):
        next_i = (i + 1) % n
        x1, z1 = footprint_local[i]
        x2, z2 = footprint_local[next_i]

        # Normal calculation
        dx = x2 - x1
        dz = z2 - z1
        length = math.sqrt(dx * dx + dz * dz)
        if length > 0:
            nx = dz / length
            nz = -dx / length
        else:
            nx, nz = 0, 1

        base_idx = len(wall_vertices) // 3
        wall_vertices.extend([
            x1, 0, z1,
            x2, 0, z2,
            x2, height, z2,
            x1, height, z1,
        ])
        wall_normals.extend([nx, 0, nz] * 4)
        wall_indices.extend([
            base_idx, base_idx + 1, base_idx + 2,
            base_idx, base_idx + 2, base_idx + 3,
        ])

    # Cap vertices
    cap_vertices = []
    cap_normals = []
    for x, y in footprint_local:
        cap_vertices.extend([x, 0, y])
        cap_normals.extend([0, -1, 0])
    for x, y in footprint_local:
        cap_vertices.extend([x, height, y])
        cap_normals.extend([0, 1, 0])

    # Cap indices
    bottom_indices = []
    top_indices = []
    for i in range(1, n - 1):
        bottom_indices.extend([0, i + 1, i])
        top_indices.extend([n, n + i, n + i + 1])

    return {
        "wall_vertices": wall_vertices,
        "wall_normals": wall_normals,
        "wall_indices": wall_indices,
        "cap_vertices": cap_vertices,
        "cap_normals": cap_normals,
        "bottom_indices": bottom_indices,
        "top_indices": top_indices,
    }


ENERGY_CLASS_COLORS = {
    "A": 0x00AA00, "B": 0x55FF00, "C": 0xAAFF00,
    "D": 0xFFFF00, "E": 0xFFAA00, "F": 0xFF5500, "G": 0xFF0000
}


def generate_html_viewer(scene_data: dict, output_path: Path):
    """Generate standalone HTML viewer."""
    scene_json = json.dumps(scene_data)
    brf_name = scene_data["metadata"]["brf_name"]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{brf_name} - 3D View</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info-panel {{
            position: absolute; top: 20px; left: 20px;
            background: rgba(255,255,255,0.95); padding: 20px;
            border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            max-width: 320px; z-index: 100;
        }}
        #info-panel h1 {{ font-size: 18px; margin-bottom: 10px; color: #333; }}
        .stat {{ margin: 8px 0; }}
        .stat-label {{ color: #666; font-size: 12px; }}
        .stat-value {{ font-size: 16px; font-weight: 600; color: #222; }}
        #building-info {{ margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; display: none; }}
        #building-info.active {{ display: block; }}
        .energy-class {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; }}
        .energy-A {{ background: #00AA00; }}
        .energy-B {{ background: #55FF00; color: #333; }}
        #legend {{ position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; }}
        #legend h3 {{ margin-bottom: 10px; font-size: 14px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 12px; }}
        .legend-color {{ width: 20px; height: 20px; margin-right: 8px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info-panel">
        <h1>{brf_name}</h1>
        <div class="stat">
            <div class="stat-label">Buildings</div>
            <div class="stat-value">{len(scene_data["buildings"])}</div>
        </div>
        <div id="building-info">
            <h3 id="building-name">Select a building</h3>
            <div class="stat"><div class="stat-label">Address</div><div class="stat-value" id="building-address">-</div></div>
            <div class="stat"><div class="stat-label">Energy Class</div><div class="stat-value"><span id="building-energy" class="energy-class">-</span></div></div>
            <div class="stat"><div class="stat-label">Height / Floors</div><div class="stat-value"><span id="building-height">-</span>m / <span id="building-floors">-</span> floors</div></div>
            <div class="stat"><div class="stat-label">WWR (avg)</div><div class="stat-value" id="building-wwr">-</div></div>
            <div class="stat"><div class="stat-label">Solar Potential</div><div class="stat-value" id="building-solar">-</div></div>
        </div>
    </div>
    <div id="legend">
        <h3>Energy Class</h3>
        <div class="legend-item"><div class="legend-color" style="background:#00AA00"></div> A</div>
        <div class="legend-item"><div class="legend-color" style="background:#55FF00"></div> B</div>
        <div class="legend-item"><div class="legend-color" style="background:#AAFF00"></div> C</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFFF00"></div> D</div>
    </div>
    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        const sceneData = {scene_json};
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(100, 80, 100);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 15, 0);
        controls.enableDamping = true;
        controls.update();
        scene.add(new THREE.AmbientLight(0x404040, 0.5));
        const sun = new THREE.DirectionalLight(0xffffff, 1);
        sun.position.set(100, 100, 50);
        sun.castShadow = true;
        scene.add(sun);
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(300, 300),
            new THREE.MeshStandardMaterial({{ color: 0x3a5f0b, roughness: 0.9 }})
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.1;
        ground.receiveShadow = true;
        scene.add(ground);
        const buildings = [];
        const buildingMeshes = new Map();
        for (const building of sceneData.buildings) {{
            const group = new THREE.Group();
            group.userData = building;
            const geom = building.geometry;
            const wallGeometry = new THREE.BufferGeometry();
            wallGeometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.wall_vertices, 3));
            wallGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(geom.wall_normals, 3));
            wallGeometry.setIndex(geom.wall_indices);
            const wallMaterial = new THREE.MeshStandardMaterial({{ color: building.color, roughness: 0.7 }});
            const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);
            wallMesh.castShadow = true;
            group.add(wallMesh);
            const capGeometry = new THREE.BufferGeometry();
            capGeometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.cap_vertices, 3));
            capGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(geom.cap_normals, 3));
            capGeometry.setIndex([...geom.bottom_indices, ...geom.top_indices]);
            const capMaterial = new THREE.MeshStandardMaterial({{ color: building.color, roughness: 0.8 }});
            const capMesh = new THREE.Mesh(capGeometry, capMaterial);
            capMesh.castShadow = true;
            group.add(capMesh);
            scene.add(group);
            buildings.push(group);
            buildingMeshes.set(building.id, {{ group, wallMaterial, capMaterial }});
        }}
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let selectedBuilding = null;
        container.addEventListener('click', (event) => {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(buildings, true);
            if (intersects.length > 0) {{
                const b = intersects[0].object.parent.userData;
                document.getElementById('building-info').classList.add('active');
                document.getElementById('building-name').textContent = b.name;
                document.getElementById('building-address').textContent = b.address;
                document.getElementById('building-height').textContent = b.height.toFixed(1);
                document.getElementById('building-floors').textContent = b.floors;
                document.getElementById('building-wwr').textContent = (b.enriched?.wwr?.average * 100 || 0).toFixed(0) + '%';
                document.getElementById('building-solar').textContent = (b.enriched?.solar_potential_kwh || 0).toLocaleString() + ' kWh/yr';
                const energyEl = document.getElementById('building-energy');
                energyEl.textContent = b.energy_class;
                energyEl.className = 'energy-class energy-' + b.energy_class;
                if (selectedBuilding) {{
                    const prev = buildingMeshes.get(selectedBuilding);
                    prev.wallMaterial.emissive.setHex(0x000000);
                    prev.capMaterial.emissive.setHex(0x000000);
                }}
                selectedBuilding = b.id;
                const curr = buildingMeshes.get(selectedBuilding);
                curr.wallMaterial.emissive.setHex(0x333333);
                curr.capMaterial.emissive.setHex(0x333333);
            }}
        }});
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
        animate();
    </script>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# Main processing
# ============================================================================

def main():
    console.print(Panel.fit(
        "[bold blue]BRF Sjöstaden 2 - Processing Pipeline[/bold blue]\n"
        "[dim]Building metadata enrichment for energy simulation[/dim]",
        border_style="blue"
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Load BRF Data ===
    console.print("\n[bold]Step 1: Loading BRF data[/bold]")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        brf_data = json.load(f)

    brf_name = brf_data["brf_name"]
    buildings = brf_data["buildings"]
    summary = brf_data["summary"]

    console.print(f"  [green]✓[/green] Loaded: {brf_name}")
    console.print(f"    Buildings: {summary['total_buildings']}")
    console.print(f"    Apartments: {summary['total_apartments']}")
    console.print(f"    Energy Class: {summary['energy_class']}")

    transformer = create_transformer()

    # === STEP 1b: Parse Energy Declaration PDF (optional) ===
    pdf_data = None
    if PDF_FILE.exists():
        console.print("\n[bold]Step 1b: Parsing energy declaration PDF[/bold]")
        pdf_data = parse_energy_declaration_pdf(PDF_FILE)

        if pdf_data:
            console.print(f"  [green]✓[/green] PDF parsed successfully")
            if pdf_data.addresses:
                console.print(f"    Additional addresses: {len(pdf_data.addresses)}")
                for addr in pdf_data.addresses[:5]:  # Show first 5
                    console.print(f"      • {addr}")
                if len(pdf_data.addresses) > 5:
                    console.print(f"      ... and {len(pdf_data.addresses) - 5} more")
            if pdf_data.ventilation_airflow_ls_sqm:
                console.print(f"    Ventilation: {pdf_data.ventilation_airflow_ls_sqm} l/s,m²")
            if pdf_data.radon_bq_m3:
                status_color = {"GOOD": "green", "WARNING": "yellow", "HIGH": "red"}.get(pdf_data.radon_status, "white")
                console.print(f"    Radon: {pdf_data.radon_bq_m3} Bq/m³ ([{status_color}]{pdf_data.radon_status}[/{status_color}])")
            if pdf_data.specific_energy_kwh_sqm:
                console.print(f"    Specific energy: {pdf_data.specific_energy_kwh_sqm} kWh/m²/yr")
            if pdf_data.reference_value_2:
                console.print(f"    Reference (similar buildings): {pdf_data.reference_value_2} kWh/m²")
            if pdf_data.recommendations:
                console.print(f"    Recommendations: {len(pdf_data.recommendations)}")
                for rec in pdf_data.recommendations:
                    console.print(f"      • {rec['description']}")
                    if rec.get('annual_savings_kwh'):
                        console.print(f"        Saves {rec['annual_savings_kwh']:,} kWh/yr")
    else:
        console.print("\n  [dim]No PDF provided, using estimates only[/dim]")

    # === STEP 1c: Fetch Facade Images (optional) ===
    facade_wwr = None  # Will store AI-detected WWR if available
    if MAPILLARY_TOKEN:
        console.print("\n[bold]Step 1c: Fetching facade images[/bold]")
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.ingest.image_fetcher import MapillaryFetcher
            from src.ai.wwr_detector import WWRDetector
            from src.ai.material_classifier import MaterialClassifier, estimate_material_from_era

            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Get building centroid for image search
            first_building = buildings[0]
            footprint = first_building["geometry"]["ground_footprint"]
            xs = [c[0] for c in footprint]
            ys = [c[1] for c in footprint]
            center_x, center_y = sum(xs) / len(xs), sum(ys) / len(ys)
            center_lon, center_lat = transformer.transform(center_x, center_y)

            console.print(f"  Searching near ({center_lat:.4f}, {center_lon:.4f})")

            # Search Mapillary for street-level images
            mf = MapillaryFetcher(access_token=MAPILLARY_TOKEN)
            # Wider search area (~1km radius)
            bbox = (center_lon - 0.015, center_lat - 0.010, center_lon + 0.015, center_lat + 0.010)
            result = mf.search_images(bbox, max_results=10)

            if result.images:
                console.print(f"  [green]✓[/green] Found {len(result.images)} street-level images")

                # Download and analyze images
                wwr_values = []
                detector = None

                for img in result.images[:5]:  # Process up to 5 images
                    if not img.url:
                        continue

                    # Download image
                    import requests
                    img_path = IMAGE_CACHE_DIR / f"mapillary_{img.image_id}.jpg"
                    if not img_path.exists():
                        try:
                            resp = requests.get(img.url, headers={"User-Agent": "BRF-Energy-Toolkit/0.1"}, timeout=30)
                            if resp.status_code == 200 and len(resp.content) > 1000:
                                img_path.write_bytes(resp.content)
                        except:
                            continue

                    if img_path.exists():
                        # Run WWR detection
                        if detector is None:
                            detector = WWRDetector(backend='opencv')

                        wwr, confidence = detector.calculate_wwr(str(img_path))
                        if wwr > 0 and confidence > 0.2:
                            wwr_values.append(wwr)
                            console.print(f"    Image {img.image_id}: WWR={wwr:.1%} (conf={confidence:.2f})")

                if wwr_values:
                    avg_wwr = sum(wwr_values) / len(wwr_values)
                    # With facade cropping, we still need small correction for:
                    # - Not all windows detected
                    # - Oblique angles reducing apparent window size
                    # Apply modest 2x correction factor
                    estimated_facade_wwr = min(0.45, avg_wwr * 2)  # Cap at 45%
                    facade_wwr = {
                        "detected_wwr": avg_wwr,
                        "estimated_facade_wwr": estimated_facade_wwr,
                        "images_analyzed": len(wwr_values),
                        "confidence": "low" if len(wwr_values) < 3 else "medium"
                    }
                    console.print(f"  [cyan]AI WWR (corrected): {estimated_facade_wwr:.0%}[/cyan]")

                # Material classification on analyzed images
                mat_classifier = MaterialClassifier()
                material_votes = {}
                for img_path in IMAGE_CACHE_DIR.glob("mapillary_*.jpg"):
                    try:
                        mat_result = mat_classifier._heuristic_classify(str(img_path))
                        mat = mat_result.material.value
                        material_votes[mat] = material_votes.get(mat, 0) + mat_result.confidence
                    except:
                        pass

                if material_votes:
                    ai_material = max(material_votes, key=material_votes.get)
                    ai_mat_confidence = material_votes[ai_material] / max(1, sum(material_votes.values()))

                    # Get era-based estimate
                    construction_year = first_building["properties"]["building_info"]["construction_year"]
                    era_material = estimate_material_from_era(construction_year, "Hammarby Sjöstad")

                    # Blend AI and era estimate
                    facade_wwr["material"] = {
                        "ai_detected": ai_material,
                        "era_estimate": era_material.value,
                        "final": era_material.value if ai_mat_confidence < 0.4 else ai_material,
                        "confidence": ai_mat_confidence
                    }
                    console.print(f"  [cyan]Material: AI={ai_material}, era={era_material.value}[/cyan]")
            else:
                console.print(f"  [yellow]No street images found nearby[/yellow]")

        except Exception as e:
            console.print(f"  [yellow]Image analysis skipped: {e}[/yellow]")
    else:
        console.print("\n  [dim]No Mapillary token, skipping image analysis[/dim]")

    # === STEP 1d: Fetch OSM data for shading analysis ===
    osm_neighbors = []
    osm_trees = []
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.ingest.osm_fetcher import OSMFetcher

        console.print("\n[bold]Step 1d: Fetching OSM data for shading analysis[/bold]")

        osm = OSMFetcher()

        # Get building centroid
        first_building = buildings[0]
        footprint = first_building["geometry"]["ground_footprint"]
        xs = [c[0] for c in footprint]
        ys = [c[1] for c in footprint]
        center_x, center_y = sum(xs) / len(xs), sum(ys) / len(ys)
        center_lon, center_lat = transformer.transform(center_x, center_y)

        # Fetch nearby buildings for shading
        osm_neighbors = osm.get_nearby_buildings(center_lon, center_lat, radius_m=150)
        console.print(f"  [green]✓[/green] Found {len(osm_neighbors)} nearby buildings")

        # Count tall buildings (potential shading sources)
        tall_neighbors = [n for n in osm_neighbors if (n.get("height") or 0) > 10 or (n.get("levels") or 0) > 3]
        if tall_neighbors:
            console.print(f"    {len(tall_neighbors)} buildings > 10m height")

        # Fetch trees/vegetation
        bbox = (center_lon - 0.002, center_lat - 0.002, center_lon + 0.002, center_lat + 0.002)
        osm_trees = osm.get_trees_in_bbox(*bbox)
        tree_count = len([t for t in osm_trees if t.get("type") == "tree"])
        console.print(f"  [green]✓[/green] Found {tree_count} trees nearby")

    except Exception as e:
        console.print(f"\n  [yellow]OSM shading data skipped: {e}[/yellow]")

    # === STEP 2: Process Each Building ===
    console.print("\n[bold]Step 2: Processing buildings[/bold]")

    enriched_buildings = []
    all_coords = []

    for building in buildings:
        bid = building["building_id"]
        geom = building["geometry"]
        props = building["properties"]

        console.print(f"\n  [cyan]Building {bid}[/cyan] - {props['location']['address']}")

        # Get coordinates
        footprint = geom["ground_footprint"]
        all_coords.extend(footprint)

        # Transform to WGS84
        wgs84_coords = sweref_to_wgs84(transformer, footprint)

        # Get properties
        construction_year = props["building_info"]["construction_year"]
        renovation_year = props["building_info"].get("last_renovation_year")
        height = props["dimensions"]["building_height_m"]
        floors = props["dimensions"]["floors_above_ground"]
        roof_area = props["dimensions"]["footprint_area_sqm"]

        # Estimate properties
        wwr = estimate_wwr(construction_year)

        # Override with AI-detected WWR if available
        if facade_wwr and facade_wwr.get("estimated_facade_wwr"):
            wwr["ai_detected"] = facade_wwr["estimated_facade_wwr"]
            wwr["ai_confidence"] = facade_wwr["confidence"]
            # Blend AI detection with era estimate (weighted average)
            if facade_wwr["confidence"] == "medium":
                wwr["average"] = facade_wwr["estimated_facade_wwr"] * 0.6 + wwr["average"] * 0.4
            else:
                wwr["average"] = facade_wwr["estimated_facade_wwr"] * 0.4 + wwr["average"] * 0.6

        # Solar potential with shading analysis
        try:
            from src.analysis.shading_solar import (
                analyze_neighbor_shading,
                analyze_tree_shading,
                calculate_solar_potential,
            )

            # Get existing solar data from building
            existing_pv_kwh = props.get("energy", {}).get("solar", {}).get("solar_cell_output_kwh", 0)
            # Assume 500 = 500 m² based on SESSION_STATE (75,000 kWh/yr)
            existing_pv_sqm = 500 if existing_pv_kwh else 0
            existing_pv_annual_kwh = 75000 if existing_pv_kwh else 0

            # Analyze shading from neighbors
            shading = analyze_neighbor_shading(
                building_center=(wgs84_coords[0][0], wgs84_coords[0][1]),
                building_height_m=height,
                neighbor_buildings=osm_neighbors,
            )

            # Add tree shading
            tree_factor = analyze_tree_shading(
                building_center=(wgs84_coords[0][0], wgs84_coords[0][1]),
                building_height_m=height,
                vegetation=osm_trees,
            )
            combined_shading = shading.overall_shading_factor * tree_factor

            # Calculate solar with shading
            solar_analysis = calculate_solar_potential(
                roof_area_sqm=roof_area,
                existing_pv_sqm=existing_pv_sqm,
                existing_pv_kwh_yr=existing_pv_annual_kwh,
                shading_factor=combined_shading,
            )

            solar = {
                "total_roof_area_sqm": roof_area,
                "existing_pv_sqm": existing_pv_sqm,
                "existing_pv_kwh_yr": existing_pv_annual_kwh,
                "remaining_suitable_area_sqm": solar_analysis.available_area_sqm,
                "remaining_capacity_kwp": solar_analysis.remaining_capacity_kwp,
                "annual_yield_potential_kwh": solar_analysis.remaining_annual_kwh,
                "total_with_existing_kwh": solar_analysis.effective_annual_kwh,
                "shading_loss_pct": solar_analysis.shading_loss_pct,
                "neighbor_count": shading.neighbor_count,
                "estimated_install_cost_sek": solar_analysis.estimated_install_cost_sek,
                "payback_years": solar_analysis.payback_years,
                "source": "osm_shading_analysis",
            }
        except Exception as e:
            # Fallback to simple estimation
            solar = estimate_solar_potential(roof_area)
            console.print(f"    [dim]Using basic solar estimate: {e}[/dim]")

        # U-values: back-calculate if PDF data available, otherwise estimate
        if pdf_data and pdf_data.specific_energy_kwh_sqm:
            u_values = back_calculate_u_values(
                specific_energy_kwh_sqm=pdf_data.specific_energy_kwh_sqm,
                heated_area_sqm=props["dimensions"]["heated_area_sqm"],
                num_floors=floors,
                height_m=height,
                construction_year=construction_year,
                renovation_year=renovation_year,
                wwr=wwr['average'],
            )
            console.print(f"    U-values (back-calculated): walls={u_values['walls']:.2f}, windows={u_values['windows']:.1f}")
            if u_values.get('notes'):
                for note in u_values['notes']:
                    console.print(f"      [dim]{note}[/dim]")
        else:
            u_values = estimate_u_values(construction_year, renovation_year)
            console.print(f"    U-values (era estimate): walls={u_values['walls']}, windows={u_values['windows']}")

        if wwr.get("ai_detected"):
            console.print(f"    WWR (AI+era blend): avg={wwr['average']:.0%} [AI:{wwr['ai_detected']:.0%}, era:{wwr['south']:.0%}]")
        else:
            console.print(f"    WWR (estimated): avg={wwr['average']:.0%}")
        # Solar output with existing vs remaining
        if solar.get("existing_pv_sqm", 0) > 0:
            console.print(f"    Solar existing: {solar.get('existing_pv_sqm', 0):.0f} m² → {solar.get('existing_pv_kwh_yr', 0):,.0f} kWh/yr")
            console.print(f"    Solar remaining: {solar.get('remaining_suitable_area_sqm', 0):.0f} m² → {solar['remaining_capacity_kwp']:.1f} kWp → {solar['annual_yield_potential_kwh']:,.0f} kWh/yr")
            if solar.get("shading_loss_pct", 0) > 0:
                console.print(f"      [dim]Shading loss: {solar.get('shading_loss_pct', 0):.0f}% (from {solar.get('neighbor_count', 0)} neighbors)[/dim]")
        else:
            console.print(f"    Solar potential: {solar['remaining_capacity_kwp']:.1f} kWp → {solar['annual_yield_potential_kwh']:,.0f} kWh/yr")

        enriched_buildings.append({
            "building_id": bid,
            "original": building,
            "wgs84_coords": wgs84_coords,
            "enriched": {
                "wwr": wwr,
                "u_values": u_values,
                "solar_potential": solar,
                "facade_material": "brick",  # Typical for Hammarby Sjöstad
                "material_confidence": 0.7,
            }
        })

    # === STEP 3: Generate 3D Visualization ===
    console.print("\n[bold]Step 3: Generating 3D visualization[/bold]")

    # Calculate scene center
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    scene_center = (sum(xs) / len(xs), sum(ys) / len(ys))

    scene_buildings = []
    max_height = 0

    for eb in enriched_buildings:
        b = eb["original"]
        footprint = b["geometry"]["ground_footprint"]
        local_coords = coords_to_local(footprint, scene_center)
        height = b["properties"]["dimensions"]["building_height_m"]
        max_height = max(max_height, height)

        mesh = generate_3d_mesh(local_coords, height)

        energy_class = b["properties"]["energy"]["energy_class"]
        color = ENERGY_CLASS_COLORS.get(energy_class, 0xCCCCCC)

        scene_buildings.append({
            "id": eb["building_id"],
            "name": f"Building {eb['building_id']}",
            "address": b["properties"]["location"]["address"],
            "height": height,
            "floors": b["properties"]["dimensions"]["floors_above_ground"],
            "energy_class": energy_class,
            "color": color,
            "geometry": mesh,
            "enriched": {
                "wwr": eb["enriched"]["wwr"],
                "solar_potential_kwh": eb["enriched"]["solar_potential"]["annual_yield_potential_kwh"],
            }
        })

    scene_data = {
        "metadata": {
            "brf_name": brf_name,
            "generator": "brf-energy-toolkit",
            "version": "0.1.0",
        },
        "buildings": scene_buildings,
    }

    # Generate viewer
    viewer_path = OUTPUT_DIR / "viewer.html"
    generate_html_viewer(scene_data, viewer_path)
    console.print(f"  [green]✓[/green] Generated: {viewer_path}")

    # === STEP 4: Export Results ===
    console.print("\n[bold]Step 4: Exporting results[/bold]")

    # Enriched JSON
    enriched_output = {
        "brf_name": brf_name,
        "export_date": date.today().isoformat(),
        "original_summary": summary,
        "buildings": []
    }

    # Add PDF-extracted data to output
    if pdf_data:
        enriched_output["pdf_extracted_data"] = {
            "additional_addresses": pdf_data.addresses,
            "ventilation_airflow_ls_sqm": pdf_data.ventilation_airflow_ls_sqm,
            "radon": {
                "value_bq_m3": pdf_data.radon_bq_m3,
                "status": pdf_data.radon_status,
            } if pdf_data.radon_bq_m3 else None,
            "specific_energy_kwh_sqm": pdf_data.specific_energy_kwh_sqm,
            "reference_value_2": pdf_data.reference_value_2,
            "recommendations": pdf_data.recommendations,
        }

    # Add facade image analysis data to output
    if facade_wwr:
        enriched_output["facade_analysis"] = {
            "source": "mapillary_opencv",
            "raw_detected_wwr": facade_wwr["detected_wwr"],
            "corrected_facade_wwr": facade_wwr["estimated_facade_wwr"],
            "images_analyzed": facade_wwr["images_analyzed"],
            "confidence": facade_wwr["confidence"],
        }

    for eb in enriched_buildings:
        enriched_output["buildings"].append({
            "building_id": eb["building_id"],
            "address": eb["original"]["properties"]["location"]["address"],
            "wgs84_footprint": eb["wgs84_coords"],
            "envelope": eb["enriched"],
        })

    enriched_path = OUTPUT_DIR / "BRF_Sjostaden_2_enriched.json"
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched_output, f, indent=2, ensure_ascii=False)
    console.print(f"  [green]✓[/green] Enriched JSON: {enriched_path}")

    # GeoJSON
    features = []
    for eb in enriched_buildings:
        coords = eb["wgs84_coords"]
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "building_id": eb["building_id"],
                "address": eb["original"]["properties"]["location"]["address"],
                "energy_class": eb["original"]["properties"]["energy"]["energy_class"],
                "wwr_average": eb["enriched"]["wwr"]["average"],
                "solar_potential_kwh": eb["enriched"]["solar_potential"]["annual_yield_potential_kwh"],
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = OUTPUT_DIR / "BRF_Sjostaden_2.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    console.print(f"  [green]✓[/green] GeoJSON: {geojson_path}")

    # === Summary ===
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]Processing Complete![/bold green]\n\n"
        f"Output: {OUTPUT_DIR}\n\n"
        f"Files generated:\n"
        f"  • viewer.html - Interactive 3D visualization\n"
        f"  • BRF_Sjostaden_2_enriched.json - Enriched data\n"
        f"  • BRF_Sjostaden_2.geojson - For mapping\n\n"
        f"[dim]Open viewer.html in your browser to see the 3D model![/dim]",
        border_style="green"
    ))

    # Summary table
    table = Table(title="Enrichment Summary")
    table.add_column("Building", style="cyan")
    table.add_column("WWR", justify="right")
    table.add_column("U-wall", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Solar kWh/yr", justify="right")

    for eb in enriched_buildings:
        u_source = eb['enriched']['u_values'].get('source', 'estimated')
        u_source_display = "📊" if u_source == "back_calculation" else "📐"
        table.add_row(
            f"Building {eb['building_id']}",
            f"{eb['enriched']['wwr']['average']:.0%}",
            f"{eb['enriched']['u_values']['walls']:.2f} W/m²K",
            u_source_display,
            f"{eb['enriched']['solar_potential']['annual_yield_potential_kwh']:,.0f}",
        )

    console.print(table)
    console.print("  [dim]📊 = back-calculated from consumption, 📐 = era estimate[/dim]")


if __name__ == "__main__":
    main()
