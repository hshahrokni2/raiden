"""
3D Building visualization generator.

Creates Three.js compatible data for web-based 3D visualization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..core.models import BRFProperty, EnrichedBRFProperty, BRFBuilding, FacadeMaterial
from ..core.coordinates import CoordinateTransformer


class Building3DGenerator:
    """
    Generate 3D visualization data for BRF buildings.

    Outputs Three.js compatible JSON for web-based rendering.
    """

    # Material colors (hex)
    MATERIAL_COLORS = {
        FacadeMaterial.BRICK: 0xB5651D,
        FacadeMaterial.CONCRETE: 0x808080,
        FacadeMaterial.PLASTER: 0xFFF8DC,
        FacadeMaterial.GLASS: 0x87CEEB,
        FacadeMaterial.METAL: 0xC0C0C0,
        FacadeMaterial.WOOD: 0xDEB887,
        FacadeMaterial.STONE: 0x696969,
        FacadeMaterial.UNKNOWN: 0xCCCCCC,
    }

    ENERGY_CLASS_COLORS = {
        "A": 0x00AA00,  # Dark green
        "B": 0x55FF00,  # Light green
        "C": 0xAAFF00,  # Yellow-green
        "D": 0xFFFF00,  # Yellow
        "E": 0xFFAA00,  # Orange
        "F": 0xFF5500,  # Red-orange
        "G": 0xFF0000,  # Red
    }

    def __init__(self):
        self.transformer = CoordinateTransformer()

    def generate_scene(
        self,
        brf: BRFProperty | EnrichedBRFProperty,
        color_by: str = "energy_class",  # or "material", "height"
        include_ground: bool = True,
        ground_size: float = 200,
    ) -> dict[str, Any]:
        """
        Generate a complete Three.js scene for the BRF property.

        Args:
            brf: BRF property (original or enriched)
            color_by: How to color buildings ("energy_class", "material", "height")
            include_ground: Whether to include a ground plane
            ground_size: Size of ground plane in meters

        Returns:
            Three.js compatible scene data
        """
        scene_data = {
            "metadata": {
                "brf_name": brf.brf_name,
                "generator": "brf-energy-toolkit",
                "version": "0.1.0",
            },
            "buildings": [],
            "ground": None,
            "camera": {},
            "lights": self._generate_lights(),
        }

        # Calculate scene center from all buildings
        all_coords = []
        for building in brf.buildings:
            if isinstance(brf, EnrichedBRFProperty):
                all_coords.extend(building.original_geometry.ground_footprint)
            else:
                all_coords.extend(building.geometry.ground_footprint)

        scene_center = self.transformer.calculate_centroid(all_coords)
        scene_center_wgs84 = self.transformer.sweref_to_wgs84(*scene_center)

        scene_data["metadata"]["center_wgs84"] = {
            "lon": scene_center_wgs84[0],
            "lat": scene_center_wgs84[1],
        }

        # Generate each building
        max_height = 0
        for building in brf.buildings:
            if isinstance(brf, EnrichedBRFProperty):
                building_data = self._generate_enriched_building(
                    building, scene_center, color_by
                )
            else:
                building_data = self._generate_building(
                    building, scene_center, color_by
                )

            scene_data["buildings"].append(building_data)
            max_height = max(max_height, building_data["height"])

        # Generate ground plane
        if include_ground:
            scene_data["ground"] = self._generate_ground(ground_size)

        # Set up camera
        scene_data["camera"] = self._generate_camera(
            max_height, ground_size
        )

        return scene_data

    def _generate_building(
        self,
        building: BRFBuilding,
        scene_center: tuple[float, float],
        color_by: str,
    ) -> dict[str, Any]:
        """Generate 3D data for a single building."""
        geom = building.geometry
        props = building.properties

        # Convert to local coordinates
        local_coords = self.transformer.coords_to_local(
            geom.ground_footprint, scene_center
        )

        # Remove duplicate closing vertex if present
        if local_coords[0] == local_coords[-1]:
            local_coords = local_coords[:-1]

        # Determine color
        if color_by == "energy_class":
            color = self.ENERGY_CLASS_COLORS.get(
                props.energy.energy_class.value, 0xCCCCCC
            )
        elif color_by == "height":
            # Gradient from blue (low) to red (high)
            normalized = min(geom.height_meters / 50, 1.0)
            color = self._height_to_color(normalized)
        else:
            color = self.MATERIAL_COLORS.get(FacadeMaterial.UNKNOWN)

        # Generate vertices and faces
        mesh_data = self._create_extruded_mesh(
            local_coords, props.dimensions.building_height_m
        )

        return {
            "id": building.building_id,
            "name": f"Building {building.building_id}",
            "address": props.location.address,
            "height": props.dimensions.building_height_m,
            "floors": props.dimensions.floors_above_ground,
            "energy_class": props.energy.energy_class.value,
            "color": color,
            "geometry": mesh_data,
            "properties": {
                "heated_area_sqm": props.dimensions.heated_area_sqm,
                "apartments": props.dimensions.apartments,
                "construction_year": props.building_info.construction_year,
                "energy_kwh_per_sqm": props.energy.energy_performance_kwh_per_sqm,
            },
        }

    def _generate_enriched_building(
        self,
        building,  # EnrichedBuilding
        scene_center: tuple[float, float],
        color_by: str,
    ) -> dict[str, Any]:
        """Generate 3D data for an enriched building."""
        geom = building.original_geometry
        props = building.original_properties
        ep_ready = building.energyplus_ready

        # Convert to local coordinates
        local_coords = self.transformer.coords_to_local(
            geom.ground_footprint, scene_center
        )

        if local_coords[0] == local_coords[-1]:
            local_coords = local_coords[:-1]

        # Determine color based on enriched data
        if color_by == "energy_class":
            color = self.ENERGY_CLASS_COLORS.get(
                props.energy.energy_class.value, 0xCCCCCC
            )
        elif color_by == "material":
            color = self.MATERIAL_COLORS.get(
                ep_ready.envelope.facade_material, 0xCCCCCC
            )
        elif color_by == "height":
            normalized = min(geom.height_meters / 50, 1.0)
            color = self._height_to_color(normalized)
        else:
            color = 0xCCCCCC

        mesh_data = self._create_extruded_mesh(
            local_coords, props.dimensions.building_height_m
        )

        # Include enriched data in output
        return {
            "id": building.building_id,
            "name": f"Building {building.building_id}",
            "address": props.location.address,
            "height": props.dimensions.building_height_m,
            "floors": props.dimensions.floors_above_ground,
            "energy_class": props.energy.energy_class.value,
            "color": color,
            "geometry": mesh_data,
            "properties": {
                "heated_area_sqm": props.dimensions.heated_area_sqm,
                "apartments": props.dimensions.apartments,
                "construction_year": props.building_info.construction_year,
                "energy_kwh_per_sqm": props.energy.energy_performance_kwh_per_sqm,
            },
            "enriched": {
                "facade_material": ep_ready.envelope.facade_material.value if ep_ready.envelope.facade_material else None,
                "wwr": ep_ready.envelope.window_to_wall_ratio.model_dump() if ep_ready.envelope.window_to_wall_ratio else None,
                "solar_potential_kwh": ep_ready.solar_potential.annual_yield_potential_kwh,
                "u_values": ep_ready.envelope.u_values.model_dump() if ep_ready.envelope.u_values else None,
            },
        }

    def _create_extruded_mesh(
        self,
        footprint: list[tuple[float, float]],
        height: float,
    ) -> dict[str, Any]:
        """
        Create extruded mesh data from 2D footprint.

        Returns vertices and indices for Three.js BufferGeometry.
        """
        n = len(footprint)

        # Vertices: bottom ring + top ring
        vertices = []
        normals = []

        # Bottom vertices (y = 0)
        for x, y in footprint:
            vertices.extend([x, 0, y])  # x, y(up), z
            normals.extend([0, -1, 0])  # Down-facing

        # Top vertices (y = height)
        for x, y in footprint:
            vertices.extend([x, height, y])
            normals.extend([0, 1, 0])  # Up-facing

        # Wall vertices (need separate for proper normals)
        wall_vertices = []
        wall_normals = []
        wall_indices = []

        for i in range(n):
            next_i = (i + 1) % n
            x1, z1 = footprint[i]
            x2, z2 = footprint[next_i]

            # Calculate wall normal
            dx = x2 - x1
            dz = z2 - z1
            length = math.sqrt(dx * dx + dz * dz)
            if length > 0:
                nx = dz / length
                nz = -dx / length
            else:
                nx, nz = 0, 1

            # Four vertices per wall segment
            base_idx = len(wall_vertices) // 3
            wall_vertices.extend([
                x1, 0, z1,        # bottom-left
                x2, 0, z2,        # bottom-right
                x2, height, z2,   # top-right
                x1, height, z1,   # top-left
            ])
            wall_normals.extend([nx, 0, nz] * 4)

            # Two triangles per wall
            wall_indices.extend([
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ])

        # Bottom face indices (fan triangulation)
        bottom_indices = []
        for i in range(1, n - 1):
            bottom_indices.extend([0, i + 1, i])  # Reversed for outward normal

        # Top face indices
        top_indices = []
        for i in range(1, n - 1):
            top_indices.extend([n, n + i, n + i + 1])

        return {
            "cap_vertices": vertices,
            "cap_normals": normals,
            "bottom_indices": bottom_indices,
            "top_indices": top_indices,
            "wall_vertices": wall_vertices,
            "wall_normals": wall_normals,
            "wall_indices": wall_indices,
        }

    def _generate_ground(self, size: float) -> dict[str, Any]:
        """Generate ground plane data."""
        half = size / 2
        return {
            "type": "plane",
            "width": size,
            "height": size,
            "color": 0x3a5f0b,  # Grass green
            "position": [0, -0.1, 0],  # Slightly below buildings
        }

    def _generate_camera(self, max_height: float, scene_size: float) -> dict[str, Any]:
        """Generate camera setup data."""
        distance = max(scene_size, max_height * 2) * 1.5
        return {
            "type": "perspective",
            "fov": 60,
            "position": [distance * 0.7, distance * 0.5, distance * 0.7],
            "target": [0, max_height / 2, 0],
            "near": 0.1,
            "far": distance * 10,
        }

    def _generate_lights(self) -> list[dict[str, Any]]:
        """Generate lighting setup."""
        return [
            {
                "type": "ambient",
                "color": 0x404040,
                "intensity": 0.5,
            },
            {
                "type": "directional",
                "color": 0xffffff,
                "intensity": 1.0,
                "position": [100, 100, 50],
                "castShadow": True,
            },
            {
                "type": "directional",
                "color": 0xffffff,
                "intensity": 0.3,
                "position": [-50, 50, -50],
            },
        ]

    @staticmethod
    def _height_to_color(normalized: float) -> int:
        """Convert normalized height (0-1) to color gradient."""
        # Blue to red gradient
        r = int(normalized * 255)
        b = int((1 - normalized) * 255)
        g = int((1 - abs(normalized - 0.5) * 2) * 128)
        return (r << 16) | (g << 8) | b

    def save_scene(
        self,
        scene_data: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save scene data to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2)

    def generate_html_viewer(
        self,
        scene_data: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """
        Generate a standalone HTML file with embedded Three.js viewer.

        The client can open this directly in their browser.
        """
        html_template = self._get_viewer_html(scene_data)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)

    def _get_viewer_html(self, scene_data: dict[str, Any]) -> str:
        """Generate HTML with embedded viewer and scene data."""
        scene_json = json.dumps(scene_data)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{scene_data["metadata"]["brf_name"]} - 3D View</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            max-width: 320px;
            z-index: 100;
        }}
        #info-panel h1 {{ font-size: 18px; margin-bottom: 10px; color: #333; }}
        #info-panel .stat {{ margin: 8px 0; }}
        #info-panel .stat-label {{ color: #666; font-size: 12px; }}
        #info-panel .stat-value {{ font-size: 16px; font-weight: 600; color: #222; }}
        #building-info {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            display: none;
        }}
        #building-info.active {{ display: block; }}
        .energy-class {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }}
        .energy-A {{ background: #00AA00; }}
        .energy-B {{ background: #55FF00; color: #333; }}
        .energy-C {{ background: #AAFF00; color: #333; }}
        .energy-D {{ background: #FFFF00; color: #333; }}
        .energy-E {{ background: #FFAA00; }}
        .energy-F {{ background: #FF5500; }}
        .energy-G {{ background: #FF0000; }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #controls label {{ margin-right: 15px; cursor: pointer; }}
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #legend h3 {{ margin-bottom: 10px; font-size: 14px; }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>

    <div id="info-panel">
        <h1>{scene_data["metadata"]["brf_name"]}</h1>
        <div class="stat">
            <div class="stat-label">Buildings</div>
            <div class="stat-value">{len(scene_data["buildings"])}</div>
        </div>
        <div id="building-info">
            <h3 id="building-name">Select a building</h3>
            <div class="stat">
                <div class="stat-label">Address</div>
                <div class="stat-value" id="building-address">-</div>
            </div>
            <div class="stat">
                <div class="stat-label">Energy Class</div>
                <div class="stat-value"><span id="building-energy" class="energy-class">-</span></div>
            </div>
            <div class="stat">
                <div class="stat-label">Height / Floors</div>
                <div class="stat-value"><span id="building-height">-</span>m / <span id="building-floors">-</span> floors</div>
            </div>
            <div class="stat">
                <div class="stat-label">Energy Use</div>
                <div class="stat-value"><span id="building-energy-use">-</span> kWh/m\xb2/year</div>
            </div>
            <div class="stat">
                <div class="stat-label">Heated Area</div>
                <div class="stat-value"><span id="building-area">-</span> m\xb2</div>
            </div>
        </div>
    </div>

    <div id="legend">
        <h3>Energy Class</h3>
        <div class="legend-item"><div class="legend-color" style="background:#00AA00"></div> A</div>
        <div class="legend-item"><div class="legend-color" style="background:#55FF00"></div> B</div>
        <div class="legend-item"><div class="legend-color" style="background:#AAFF00"></div> C</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFFF00"></div> D</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFAA00"></div> E</div>
        <div class="legend-item"><div class="legend-color" style="background:#FF5500"></div> F</div>
        <div class="legend-item"><div class="legend-color" style="background:#FF0000"></div> G</div>
    </div>

    <div id="controls">
        <strong>Color by:</strong>
        <label><input type="radio" name="colorBy" value="energy_class" checked> Energy Class</label>
        <label><input type="radio" name="colorBy" value="height"> Height</label>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const sceneData = {scene_json};

        // Setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);

        const camera = new THREE.PerspectiveCamera(
            sceneData.camera.fov,
            window.innerWidth / window.innerHeight,
            sceneData.camera.near,
            sceneData.camera.far
        );
        camera.position.set(...sceneData.camera.position);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(...sceneData.camera.target);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.update();

        // Lights
        for (const light of sceneData.lights) {{
            let l;
            if (light.type === 'ambient') {{
                l = new THREE.AmbientLight(light.color, light.intensity);
            }} else if (light.type === 'directional') {{
                l = new THREE.DirectionalLight(light.color, light.intensity);
                l.position.set(...light.position);
                if (light.castShadow) {{
                    l.castShadow = true;
                    l.shadow.mapSize.width = 2048;
                    l.shadow.mapSize.height = 2048;
                    l.shadow.camera.near = 0.5;
                    l.shadow.camera.far = 500;
                    l.shadow.camera.left = -100;
                    l.shadow.camera.right = 100;
                    l.shadow.camera.top = 100;
                    l.shadow.camera.bottom = -100;
                }}
            }}
            if (l) scene.add(l);
        }}

        // Ground
        if (sceneData.ground) {{
            const groundGeom = new THREE.PlaneGeometry(sceneData.ground.width, sceneData.ground.height);
            const groundMat = new THREE.MeshStandardMaterial({{
                color: sceneData.ground.color,
                roughness: 0.9
            }});
            const ground = new THREE.Mesh(groundGeom, groundMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.set(...sceneData.ground.position);
            ground.receiveShadow = true;
            scene.add(ground);
        }}

        // Buildings
        const buildings = [];
        const buildingMeshes = new Map();

        for (const building of sceneData.buildings) {{
            const group = new THREE.Group();
            group.userData = building;

            const geom = building.geometry;

            // Create wall geometry
            const wallGeometry = new THREE.BufferGeometry();
            wallGeometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.wall_vertices, 3));
            wallGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(geom.wall_normals, 3));
            wallGeometry.setIndex(geom.wall_indices);

            const wallMaterial = new THREE.MeshStandardMaterial({{
                color: building.color,
                roughness: 0.7,
                metalness: 0.1
            }});

            const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);
            wallMesh.castShadow = true;
            wallMesh.receiveShadow = true;
            group.add(wallMesh);

            // Create cap geometry (top and bottom)
            const capGeometry = new THREE.BufferGeometry();
            capGeometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.cap_vertices, 3));
            capGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(geom.cap_normals, 3));

            // Combine indices
            const allCapIndices = [...geom.bottom_indices, ...geom.top_indices];
            capGeometry.setIndex(allCapIndices);

            const capMaterial = new THREE.MeshStandardMaterial({{
                color: building.color,
                roughness: 0.8,
                metalness: 0.1
            }});

            const capMesh = new THREE.Mesh(capGeometry, capMaterial);
            capMesh.castShadow = true;
            capMesh.receiveShadow = true;
            group.add(capMesh);

            scene.add(group);
            buildings.push(group);
            buildingMeshes.set(building.id, {{ group, wallMaterial, capMaterial }});
        }}

        // Raycaster for selection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let selectedBuilding = null;

        function onMouseClick(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            const intersects = raycaster.intersectObjects(buildings, true);

            if (intersects.length > 0) {{
                const clickedGroup = intersects[0].object.parent;
                const buildingData = clickedGroup.userData;

                // Update info panel
                document.getElementById('building-info').classList.add('active');
                document.getElementById('building-name').textContent = buildingData.name;
                document.getElementById('building-address').textContent = buildingData.address;
                document.getElementById('building-height').textContent = buildingData.height.toFixed(1);
                document.getElementById('building-floors').textContent = buildingData.floors;
                document.getElementById('building-area').textContent = buildingData.properties.heated_area_sqm.toLocaleString();
                document.getElementById('building-energy-use').textContent = buildingData.properties.energy_kwh_per_sqm;

                const energyEl = document.getElementById('building-energy');
                energyEl.textContent = buildingData.energy_class;
                energyEl.className = 'energy-class energy-' + buildingData.energy_class;

                // Highlight
                if (selectedBuilding) {{
                    const prev = buildingMeshes.get(selectedBuilding);
                    prev.wallMaterial.emissive.setHex(0x000000);
                    prev.capMaterial.emissive.setHex(0x000000);
                }}
                selectedBuilding = buildingData.id;
                const curr = buildingMeshes.get(selectedBuilding);
                curr.wallMaterial.emissive.setHex(0x333333);
                curr.capMaterial.emissive.setHex(0x333333);
            }}
        }}

        container.addEventListener('click', onMouseClick);

        // Color mode switching
        const energyColors = {{
            'A': 0x00AA00, 'B': 0x55FF00, 'C': 0xAAFF00,
            'D': 0xFFFF00, 'E': 0xFFAA00, 'F': 0xFF5500, 'G': 0xFF0000
        }};

        document.querySelectorAll('input[name="colorBy"]').forEach(radio => {{
            radio.addEventListener('change', (e) => {{
                const mode = e.target.value;

                for (const building of sceneData.buildings) {{
                    const meshes = buildingMeshes.get(building.id);
                    let color;

                    if (mode === 'energy_class') {{
                        color = energyColors[building.energy_class] || 0xCCCCCC;
                    }} else if (mode === 'height') {{
                        const normalized = Math.min(building.height / 50, 1);
                        const r = Math.floor(normalized * 255);
                        const b = Math.floor((1 - normalized) * 255);
                        const g = Math.floor((1 - Math.abs(normalized - 0.5) * 2) * 128);
                        color = (r << 16) | (g << 8) | b;
                    }}

                    meshes.wallMaterial.color.setHex(color);
                    meshes.capMaterial.color.setHex(color);
                }}
            }});
        }});

        // Resize handler
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>'''
