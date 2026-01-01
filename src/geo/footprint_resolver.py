"""
Footprint Resolver for Raiden

Resolves building footprints from existing data sources (OSM, Microsoft).
Satellite extraction is LAST RESORT only.

Key features:
- Point-in-polygon lookup from OSM
- Courtyard/complex detection
- Multi-building BRF support via street addresses
- Address-based slicing of shared complexes

Usage:
    resolver = FootprintResolver()
    
    # Simple case: single building
    footprints = resolver.resolve(lat=59.37, lon=17.98)
    
    # Multi-building BRF with addresses from energy declaration
    footprints = resolver.resolve_by_addresses(
        addresses=["Storgatan 1", "Storgatan 3", "Storgatan 5"],
        city="Stockholm"
    )
"""

import requests
import math
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ResolvedFootprint:
    """A resolved building footprint with metadata."""
    geometry: Optional[dict]         # GeoJSON Polygon, or None if no building
    height_m: Optional[float] = None
    floors: Optional[int] = None
    source: str = "unknown"          # 'osm', 'microsoft', 'satellite', 'osm+address_estimation'
    osm_id: Optional[str] = None
    confidence: float = 0.0
    address: Optional[str] = None    # Street address if known
    is_complex: bool = False         # True if part of multi-building property
    buildings_in_complex: int = 1    # Number of buildings in complex
    is_estimated_boundary: bool = False  # True if boundary was estimated from addresses
    boundary_estimation_note: Optional[str] = None  # Explanation of estimation
    
    def to_dict(self) -> dict:
        return {
            "geometry": self.geometry,
            "height_m": self.height_m,
            "floors": self.floors,
            "source": self.source,
            "osm_id": self.osm_id,
            "confidence": self.confidence,
            "address": self.address,
            "is_complex": self.is_complex,
            "buildings_in_complex": self.buildings_in_complex,
            "is_estimated_boundary": self.is_estimated_boundary,
            "boundary_estimation_note": self.boundary_estimation_note,
        }


class FootprintResolver:
    """
    Resolves building footprints from existing data sources.
    
    Priority:
    1. OSM buildings (most accurate, community-maintained)
    2. Microsoft Building Footprints (AI-derived, good coverage)
    3. Satellite extraction (last resort, for new construction)
    
    Handles edge cases:
    - Courtyards (point surrounded by buildings)
    - Multi-building BRFs (using street addresses)
    - Shared complexes (slicing by address)
    - Point on street (nearest building with lower confidence)
    """
    
    OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    OSM_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Raiden/1.0 (Building Analysis Tool)"
        })
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def resolve(
        self,
        lat: float,
        lon: float,
        search_radius_m: float = 50,
    ) -> List[ResolvedFootprint]:
        """
        Resolve building footprint(s) at given coordinates.
        
        Args:
            lat, lon: Coordinates (typically from geocoded address)
            search_radius_m: Search radius for nearby buildings
            
        Returns:
            List of ResolvedFootprint objects. May contain multiple
            buildings if point is in a courtyard surrounded by buildings.
        """
        # === TIER 1: Exact point-in-polygon ===
        building = self._osm_building_at_point(lat, lon)
        if building:
            return [building]
        
        # === TIER 2: Courtyard/complex detection ===
        all_nearby = self._osm_buildings_in_radius(lat, lon, search_radius_m)
        
        if len(all_nearby) >= 2:
            # Check if buildings form a courtyard around the point
            if self._is_courtyard_scenario(all_nearby, lat, lon):
                return self._convert_to_complex(all_nearby)
            else:
                # Not a courtyard - return nearest with lower confidence
                return [self._find_nearest_building(all_nearby, lat, lon)]
        
        elif len(all_nearby) == 1:
            distance = self._distance_to_building(all_nearby[0], lat, lon)
            confidence = self._confidence_by_distance(distance)
            return [self._osm_element_to_footprint(all_nearby[0], confidence=confidence)]
        
        # === TIER 3: No buildings found ===
        return []
    
    def resolve_by_addresses(
        self,
        addresses: List[str],
        city: str,
        country: str = "Sweden",
    ) -> List[ResolvedFootprint]:
        """
        Resolve footprints for a list of street addresses.
        
        This is the BEST method for Swedish BRFs because energy declarations
        contain the actual street addresses of buildings in the BRF.
        
        Args:
            addresses: List of street addresses (e.g., ["Storgatan 1", "Storgatan 3"])
            city: City name (e.g., "Stockholm", "Solna")
            country: Country (default: "Sweden")
            
        Returns:
            List of ResolvedFootprint objects, one per successfully resolved address.
            
        Example:
            # BRF owns buildings at these addresses
            footprints = resolver.resolve_by_addresses(
                addresses=["Filmgatan 1", "Filmgatan 3", "Filmgatan 5"],
                city="Solna"
            )
        """
        results = []
        
        for address in addresses:
            full_address = f"{address}, {city}, {country}"
            
            # Geocode the address
            coords = self._geocode_address(full_address)
            if not coords:
                print(f"[FootprintResolver] Could not geocode: {full_address}")
                continue
            
            lat, lon = coords
            
            # Find building at this address
            building = self._osm_building_at_point(lat, lon)
            
            if not building:
                # Try nearby buildings and match by address
                nearby = self._osm_buildings_in_radius(lat, lon, radius_m=30)
                building = self._find_building_by_address(nearby, address)
            
            if not building:
                # Last resort: nearest building
                nearby = self._osm_buildings_in_radius(lat, lon, radius_m=50)
                if nearby:
                    building = self._find_nearest_building(nearby, lat, lon)
            
            if building:
                building.address = address
                building.is_complex = len(addresses) > 1
                building.buildings_in_complex = len(addresses)
                results.append(building)
        
        return results
    
    def resolve_with_address_slicing(
        self,
        lat: float,
        lon: float,
        brf_addresses: List[str],
        search_radius_m: float = 100,
    ) -> List[ResolvedFootprint]:
        """
        Resolve footprints for a BRF, using addresses to slice shared complexes.
        
        This handles the case where multiple BRFs share a large complex:
        - Query all buildings in radius
        - Filter to only buildings matching BRF's addresses
        - Return only those buildings
        
        Args:
            lat, lon: Center coordinates (e.g., BRF's primary location)
            brf_addresses: List of addresses from energy declaration
            search_radius_m: Search radius
            
        Returns:
            List of footprints for buildings matching the BRF's addresses.
        """
        if not brf_addresses:
            # No addresses provided - fall back to standard resolution
            return self.resolve(lat, lon, search_radius_m)
        
        # Get all buildings in radius
        all_buildings = self._osm_buildings_in_radius(lat, lon, search_radius_m)
        
        if not all_buildings:
            # No buildings found - try address-based resolution
            return self.resolve_by_addresses(
                brf_addresses, 
                city=self._guess_city_from_coords(lat, lon)
            )
        
        # Normalize BRF addresses for matching
        normalized_brf_addresses = {
            self._normalize_address(addr) for addr in brf_addresses
        }
        
        # Filter buildings to those matching BRF's addresses
        matched_buildings = []
        for building in all_buildings:
            building_addr = self._get_building_address(building)
            if building_addr:
                normalized = self._normalize_address(building_addr)
                if normalized in normalized_brf_addresses:
                    matched_buildings.append(building)
                elif self._address_fuzzy_match(normalized, normalized_brf_addresses):
                    matched_buildings.append(building)
        
        if matched_buildings:
            results = []
            for building in matched_buildings:
                fp = self._osm_element_to_footprint(building, confidence=0.90)
                fp.address = self._get_building_address(building)
                fp.is_complex = len(matched_buildings) > 1
                fp.buildings_in_complex = len(matched_buildings)
                results.append(fp)
            return results
        
        # No address matches - fall back to nearest building
        return self.resolve(lat, lon, search_radius_m)
    
    # =========================================================================
    # INTERNAL: OSM QUERIES
    # =========================================================================
    
    def _osm_building_at_point(self, lat: float, lon: float) -> Optional[ResolvedFootprint]:
        """Query OSM for building containing this exact point."""
        # Query for both buildings and building:part (for mixed-use complexes)
        query = f"""
        [out:json][timeout:10];
        (
            way["building"](around:1,{lat},{lon});
            way["building:part"](around:1,{lat},{lon});
        );
        out geom;
        """
        buildings = self._execute_osm_query(query)
        
        if not buildings:
            return None
        
        # Prefer building:part (more specific for mixed-use)
        building_parts = [b for b in buildings if 'building:part' in b.get('tags', {})]
        if building_parts:
            return self._osm_element_to_footprint(building_parts[0], confidence=0.95)
        
        return self._osm_element_to_footprint(buildings[0], confidence=0.95)
    
    def _osm_buildings_in_radius(self, lat: float, lon: float, radius_m: float) -> List[dict]:
        """Get all OSM buildings within radius."""
        query = f"""
        [out:json][timeout:10];
        way["building"](around:{radius_m},{lat},{lon});
        out geom;
        """
        return self._execute_osm_query(query)
    
    def _execute_osm_query(self, query: str) -> List[dict]:
        """Execute Overpass API query."""
        try:
            resp = self._session.post(
                self.OSM_OVERPASS_URL,
                data={"data": query},
                timeout=15
            )
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as e:
            print(f"[FootprintResolver] OSM query failed: {e}")
            return []
    
    def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address using Nominatim."""
        try:
            resp = self._session.get(
                self.OSM_NOMINATIM_URL,
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                },
                timeout=10
            )
            resp.raise_for_status()
            results = resp.json()
            
            if results:
                return (float(results[0]["lat"]), float(results[0]["lon"]))
        except Exception as e:
            print(f"[FootprintResolver] Geocoding failed: {e}")
        
        return None
    
    # =========================================================================
    # INTERNAL: BUILDING MATCHING
    # =========================================================================
    
    def _find_building_by_address(
        self, 
        buildings: List[dict], 
        target_address: str
    ) -> Optional[ResolvedFootprint]:
        """Find building matching a specific address."""
        normalized_target = self._normalize_address(target_address)
        
        for building in buildings:
            building_addr = self._get_building_address(building)
            if building_addr:
                if self._normalize_address(building_addr) == normalized_target:
                    fp = self._osm_element_to_footprint(building, confidence=0.92)
                    fp.address = building_addr
                    return fp
        
        # Fuzzy match
        for building in buildings:
            building_addr = self._get_building_address(building)
            if building_addr:
                if self._address_fuzzy_match(
                    self._normalize_address(building_addr), 
                    {normalized_target}
                ):
                    fp = self._osm_element_to_footprint(building, confidence=0.85)
                    fp.address = building_addr
                    return fp
        
        return None
    
    def _find_nearest_building(
        self, 
        buildings: List[dict], 
        lat: float, 
        lon: float,
    ) -> ResolvedFootprint:
        """Find nearest building with distance-based confidence."""
        nearest = min(buildings, key=lambda b: self._distance_to_building(b, lat, lon))
        distance = self._distance_to_building(nearest, lat, lon)
        confidence = self._confidence_by_distance(distance) * 0.8  # Lower base for "nearest"
        return self._osm_element_to_footprint(nearest, confidence=confidence)
    
    def _get_building_address(self, building: dict) -> Optional[str]:
        """Extract street address from OSM building tags."""
        tags = building.get('tags', {})
        
        street = tags.get('addr:street', '')
        housenumber = tags.get('addr:housenumber', '')
        
        if street and housenumber:
            return f"{street} {housenumber}"
        elif tags.get('addr:full'):
            return tags['addr:full']
        
        return None
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for comparison."""
        if not address:
            return ""
        
        # Lowercase
        normalized = address.lower().strip()
        
        # Swedish normalization
        normalized = normalized.replace('gatan', 'g')
        normalized = normalized.replace('vägen', 'v')
        normalized = normalized.replace('väg', 'v')
        normalized = normalized.replace('allén', 'allé')
        
        # Remove common suffixes
        normalized = re.sub(r'\s*,\s*(stockholm|solna|sundbyberg|nacka).*$', '', normalized)
        
        # Normalize spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _address_fuzzy_match(self, address: str, target_set: set) -> bool:
        """Fuzzy match address against a set of target addresses."""
        if not address or not target_set:
            return False
        
        # Extract street name and number
        match = re.match(r'^(.+?)\s*(\d+)\s*([a-z]?)$', address)
        if not match:
            return False
        
        street, number, suffix = match.groups()
        
        for target in target_set:
            target_match = re.match(r'^(.+?)\s*(\d+)\s*([a-z]?)$', target)
            if target_match:
                t_street, t_number, t_suffix = target_match.groups()
                
                # Street names similar and numbers match
                if self._string_similarity(street, t_street) > 0.8 and number == t_number:
                    return True
        
        return False
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard on character bigrams)."""
        if not s1 or not s2:
            return 0.0
        
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        b1, b2 = bigrams(s1), bigrams(s2)
        if not b1 or not b2:
            return 1.0 if s1 == s2 else 0.0
        
        intersection = len(b1 & b2)
        union = len(b1 | b2)
        return intersection / union if union else 0.0
    
    # =========================================================================
    # INTERNAL: COURTYARD DETECTION
    # =========================================================================
    
    def _is_courtyard_scenario(
        self, 
        buildings: List[dict], 
        center_lat: float, 
        center_lon: float
    ) -> bool:
        """
        Detect if buildings form a courtyard around the center point.
        
        A courtyard is when buildings surround a central open space.
        We detect this by checking if buildings are spread around the point
        (angle coverage > 180°).
        """
        if len(buildings) < 2:
            return False
        
        # Calculate angle from center to each building centroid
        angles = []
        for building in buildings:
            centroid = self._get_building_centroid(building)
            if centroid:
                angle = math.atan2(
                    centroid[0] - center_lat, 
                    centroid[1] - center_lon
                )
                angles.append(math.degrees(angle))
        
        if len(angles) < 2:
            return False
        
        # Find largest gap between buildings
        angles.sort()
        max_gap = 0
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i]
            if gap < 0:
                gap += 360
            max_gap = max(max_gap, gap)
        
        # If largest gap < 200°, buildings likely surround the point
        return max_gap < 200
    
    def _convert_to_complex(self, buildings: List[dict]) -> List[ResolvedFootprint]:
        """Convert list of buildings to complex footprints."""
        results = []
        for building in buildings:
            fp = self._osm_element_to_footprint(building, confidence=0.85)
            fp.is_complex = True
            fp.buildings_in_complex = len(buildings)
            fp.address = self._get_building_address(building)
            results.append(fp)
        return results
    
    # =========================================================================
    # INTERNAL: GEOMETRY HELPERS
    # =========================================================================
    
    def _osm_element_to_footprint(
        self, 
        element: dict, 
        confidence: float
    ) -> ResolvedFootprint:
        """Convert OSM element to ResolvedFootprint."""
        geometry = element.get('geometry', [])
        coords = [[p['lon'], p['lat']] for p in geometry]
        
        # Close polygon if needed
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        
        tags = element.get('tags', {})
        
        # Extract height
        height = None
        height_str = tags.get('height') or tags.get('building:height')
        if height_str:
            try:
                height = float(str(height_str).replace('m', '').strip())
            except:
                pass
        
        # Extract/estimate floors
        floors = None
        levels_str = tags.get('building:levels')
        if levels_str:
            try:
                floors = int(float(levels_str))
                if not height:
                    height = floors * 3.0  # ~3m per floor
            except:
                pass
        
        return ResolvedFootprint(
            geometry={"type": "Polygon", "coordinates": [coords]} if coords else None,
            height_m=height,
            floors=floors,
            source="osm",
            osm_id=str(element.get('id')),
            confidence=confidence,
            address=self._get_building_address(element),
        )
    
    def _get_building_centroid(self, building: dict) -> Optional[Tuple[float, float]]:
        """Get centroid of building polygon."""
        geom = building.get('geometry', [])
        if not geom:
            return None
        lat = sum(p['lat'] for p in geom) / len(geom)
        lon = sum(p['lon'] for p in geom) / len(geom)
        return (lat, lon)
    
    def _distance_to_building(self, building: dict, lat: float, lon: float) -> float:
        """Distance from point to building centroid in meters."""
        centroid = self._get_building_centroid(building)
        if not centroid:
            return float('inf')
        return self._haversine_distance(lat, lon, centroid[0], centroid[1])
    
    def _confidence_by_distance(self, distance_m: float) -> float:
        """Confidence degrades with distance from point."""
        if distance_m < 5:
            return 1.0
        elif distance_m < 15:
            return 0.90
        elif distance_m < 30:
            return 0.75
        elif distance_m < 50:
            return 0.60
        else:
            return 0.45
    
    def _haversine_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        """Distance in meters between two points."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (math.sin(dphi/2)**2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def _guess_city_from_coords(self, lat: float, lon: float) -> str:
        """Guess city from coordinates (rough, for fallback only)."""
        # Stockholm area
        if 59.2 < lat < 59.5 and 17.8 < lon < 18.3:
            if lon < 18.0:
                return "Solna"
            return "Stockholm"
        return "Stockholm"  # Default
    
    # =========================================================================
    # SMART BOUNDARY ESTIMATION
    # =========================================================================
    
    def resolve_with_smart_boundary(
        self,
        lat: float,
        lon: float,
        brf_addresses: List[str],
        city: str = "Stockholm",
    ) -> List[ResolvedFootprint]:
        """
        Resolve footprint with smart boundary estimation.
        
        If addresses only cover PART of a building, estimates the property 
        boundary by cutting the building at the edge of known addresses.
        
        This handles the case where two BRFs share one physical building,
        split by an internal wall.
        
        Example:
            ┌─────────────────────────────────────────┐
            │   ???        │  Storg 5   Storg 7       │
            │   (unknown)  │    ●         ●          │
            │   BRF A      │        BRF B            │
            └─────────────────────────────────────────┘
            
            If we only have addresses for BRF B (Storgatan 5, 7),
            we cut the building slightly LEFT of Storgatan 5.
        
        Args:
            lat, lon: Approximate center coordinates
            brf_addresses: List of street addresses from energy declaration
            city: City name for geocoding
            
        Returns:
            List with single ResolvedFootprint, potentially clipped.
        """
        # Get full building from OSM
        full_building = self._osm_building_at_point(lat, lon)
        
        if not full_building or not full_building.geometry:
            # Try nearby
            nearby = self._osm_buildings_in_radius(lat, lon, radius_m=50)
            if nearby:
                full_building = self._find_nearest_building(nearby, lat, lon)
            else:
                return []
        
        if not full_building or not full_building.geometry:
            return []
        
        # Geocode all BRF addresses to get points
        address_points = []
        for addr in brf_addresses:
            full_addr = f"{addr}, {city}, Sweden"
            coords = self._geocode_address(full_addr)
            if coords:
                address_points.append(coords)
        
        if not address_points:
            # Can't estimate, return full building
            full_building.boundary_estimation_note = "No addresses could be geocoded"
            return [full_building]
        
        # Estimate boundary
        clipped_geom, confidence, note = self._estimate_property_boundary(
            full_building.geometry,
            address_points
        )
        
        return [ResolvedFootprint(
            geometry=clipped_geom,
            height_m=full_building.height_m,
            floors=full_building.floors,
            source="osm+address_estimation" if note else "osm",
            osm_id=full_building.osm_id,
            confidence=confidence,
            is_estimated_boundary=bool(note),
            boundary_estimation_note=note,
        )]
    
    def _estimate_property_boundary(
        self,
        building_polygon: dict,
        address_points: List[Tuple[float, float]],
    ) -> Tuple[dict, float, Optional[str]]:
        """
        Estimate property boundary using address positions.
        
        Logic:
        1. Find bounding box of address points within building
        2. If addresses only cover PART of building, cut at the edge
        3. Add 5% buffer to avoid cutting through our property
        
        Args:
            building_polygon: GeoJSON polygon of full building
            address_points: List of (lat, lon) tuples from geocoded addresses
            
        Returns:
            (clipped_geometry, confidence, note)
            - note is None if no cutting was done
        """
        coords = building_polygon.get("coordinates", [[]])[0]
        if len(coords) < 3:
            return building_polygon, 0.5, None
        
        # Get building bounds
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        minx, maxx = min(lons), max(lons)
        miny, maxy = min(lats), max(lats)
        
        # Get address positions
        addr_lons = [p[1] for p in address_points]  # lon
        addr_lats = [p[0] for p in address_points]  # lat
        
        addr_min_lon, addr_max_lon = min(addr_lons), max(addr_lons)
        addr_min_lat, addr_max_lat = min(addr_lats), max(addr_lats)
        
        # Building dimensions
        building_width = maxx - minx   # lon span (E-W)
        building_height = maxy - miny  # lat span (N-S)
        
        if building_width == 0 or building_height == 0:
            return building_polygon, 0.5, None
        
        is_horizontal = building_width > building_height
        
        # Calculate address coverage ratio
        if is_horizontal:
            start_ratio = (addr_min_lon - minx) / building_width
            end_ratio = (addr_max_lon - minx) / building_width
        else:
            start_ratio = (addr_min_lat - miny) / building_height
            end_ratio = (addr_max_lat - miny) / building_height
        
        coverage = end_ratio - start_ratio
        
        # Decision logic
        BUFFER = 0.05  # 5% buffer
        
        if coverage > 0.75:
            # Addresses span 75%+ of building → probably own it all
            return building_polygon, 0.90, None
        
        if coverage < 0.15:
            # Very small coverage → something weird, return full with low confidence
            return building_polygon, 0.40, "Addresses cover <15% of building - unclear boundary"
        
        # Partial coverage → CUT the building!
        note = None
        clipped_coords = None
        
        if is_horizontal:
            if start_ratio > 0.25:
                # Addresses on RIGHT side → cut off the LEFT
                cut_ratio = max(0, start_ratio - BUFFER)
                cut_x = minx + cut_ratio * building_width
                clipped_coords = self._clip_polygon_left(coords, cut_x)
                note = f"Building cut at {cut_ratio*100:.0f}% from left (addresses start at {start_ratio*100:.0f}%)"
                
            elif end_ratio < 0.75:
                # Addresses on LEFT side → cut off the RIGHT
                cut_ratio = min(1, end_ratio + BUFFER)
                cut_x = minx + cut_ratio * building_width
                clipped_coords = self._clip_polygon_right(coords, cut_x)
                note = f"Building cut at {cut_ratio*100:.0f}% from left (addresses end at {end_ratio*100:.0f}%)"
        else:
            # N-S building
            if start_ratio > 0.25:
                # Addresses on NORTH side → cut off the SOUTH
                cut_ratio = max(0, start_ratio - BUFFER)
                cut_y = miny + cut_ratio * building_height
                clipped_coords = self._clip_polygon_bottom(coords, cut_y)
                note = f"Building cut at {cut_ratio*100:.0f}% from south (addresses start at {start_ratio*100:.0f}%)"
                
            elif end_ratio < 0.75:
                # Addresses on SOUTH side → cut off the NORTH
                cut_ratio = min(1, end_ratio + BUFFER)
                cut_y = miny + cut_ratio * building_height
                clipped_coords = self._clip_polygon_top(coords, cut_y)
                note = f"Building cut at {cut_ratio*100:.0f}% from south (addresses end at {end_ratio*100:.0f}%)"
        
        if clipped_coords and len(clipped_coords) >= 3:
            # Close polygon if needed
            if clipped_coords[0] != clipped_coords[-1]:
                clipped_coords.append(clipped_coords[0])
            
            confidence = 0.65 + (coverage * 0.25)  # 65-90% based on coverage
            return {"type": "Polygon", "coordinates": [clipped_coords]}, confidence, note
        
        # Couldn't clip, return original
        return building_polygon, 0.60, "Could not estimate boundary - addresses in middle of building"
    
    def _clip_polygon_left(self, coords: List, cut_x: float) -> List:
        """Clip polygon, keeping everything to the RIGHT of cut_x."""
        clipped = []
        for i in range(len(coords)):
            curr = coords[i]
            next_pt = coords[(i + 1) % len(coords)]
            
            curr_in = curr[0] >= cut_x
            next_in = next_pt[0] >= cut_x
            
            if curr_in:
                clipped.append(curr)
            
            if curr_in != next_in:
                # Edge crosses boundary
                t = (cut_x - curr[0]) / (next_pt[0] - curr[0]) if next_pt[0] != curr[0] else 0
                intersection = [cut_x, curr[1] + t * (next_pt[1] - curr[1])]
                clipped.append(intersection)
        
        return clipped
    
    def _clip_polygon_right(self, coords: List, cut_x: float) -> List:
        """Clip polygon, keeping everything to the LEFT of cut_x."""
        clipped = []
        for i in range(len(coords)):
            curr = coords[i]
            next_pt = coords[(i + 1) % len(coords)]
            
            curr_in = curr[0] <= cut_x
            next_in = next_pt[0] <= cut_x
            
            if curr_in:
                clipped.append(curr)
            
            if curr_in != next_in:
                t = (cut_x - curr[0]) / (next_pt[0] - curr[0]) if next_pt[0] != curr[0] else 0
                intersection = [cut_x, curr[1] + t * (next_pt[1] - curr[1])]
                clipped.append(intersection)
        
        return clipped
    
    def _clip_polygon_bottom(self, coords: List, cut_y: float) -> List:
        """Clip polygon, keeping everything ABOVE cut_y."""
        clipped = []
        for i in range(len(coords)):
            curr = coords[i]
            next_pt = coords[(i + 1) % len(coords)]
            
            curr_in = curr[1] >= cut_y
            next_in = next_pt[1] >= cut_y
            
            if curr_in:
                clipped.append(curr)
            
            if curr_in != next_in:
                t = (cut_y - curr[1]) / (next_pt[1] - curr[1]) if next_pt[1] != curr[1] else 0
                intersection = [curr[0] + t * (next_pt[0] - curr[0]), cut_y]
                clipped.append(intersection)
        
        return clipped
    
    def _clip_polygon_top(self, coords: List, cut_y: float) -> List:
        """Clip polygon, keeping everything BELOW cut_y."""
        clipped = []
        for i in range(len(coords)):
            curr = coords[i]
            next_pt = coords[(i + 1) % len(coords)]
            
            curr_in = curr[1] <= cut_y
            next_in = next_pt[1] <= cut_y
            
            if curr_in:
                clipped.append(curr)
            
            if curr_in != next_in:
                t = (cut_y - curr[1]) / (next_pt[1] - curr[1]) if next_pt[1] != curr[1] else 0
                intersection = [curr[0] + t * (next_pt[0] - curr[0]), cut_y]
                clipped.append(intersection)
        
        return clipped


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def resolve_footprint(lat: float, lon: float) -> List[dict]:
    """
    Quick function to resolve footprint at coordinates.
    
    Returns list of footprint dicts.
    """
    resolver = FootprintResolver()
    footprints = resolver.resolve(lat, lon)
    return [fp.to_dict() for fp in footprints]


def resolve_brf_footprints(
    lat: float, 
    lon: float, 
    addresses: List[str] = None,
    city: str = "Stockholm",
    smart_boundary: bool = True,
) -> List[dict]:
    """
    Resolve footprints for a BRF.
    
    Args:
        lat, lon: BRF coordinates
        addresses: List of street addresses from energy declaration (optional)
        city: City name
        smart_boundary: If True, uses smart boundary estimation when addresses
                       only cover part of a building (handles shared buildings)
        
    Returns:
        List of footprint dicts for all buildings in the BRF.
    """
    resolver = FootprintResolver()
    
    if addresses:
        if smart_boundary:
            # Use smart boundary estimation (handles split buildings)
            footprints = resolver.resolve_with_smart_boundary(
                lat, lon, addresses, city=city
            )
        else:
            # Simple address slicing
            footprints = resolver.resolve_with_address_slicing(
                lat, lon, addresses, search_radius_m=100
            )
    else:
        footprints = resolver.resolve(lat, lon)
    
    return [fp.to_dict() for fp in footprints]


def resolve_shared_building(
    lat: float,
    lon: float,
    brf_addresses: List[str],
    city: str = "Stockholm",
) -> dict:
    """
    Resolve footprint for a BRF that may share a building with another property.
    
    Uses smart boundary estimation based on address positions to cut the
    building at the estimated property boundary.
    
    Args:
        lat, lon: Approximate building coordinates
        brf_addresses: Street addresses belonging to this BRF
        city: City name
        
    Returns:
        Single footprint dict with estimated boundary.
        
    Example:
        # BRF owns right side of building at Storgatan 5-9
        result = resolve_shared_building(
            lat=59.33, lon=18.05,
            brf_addresses=["Storgatan 5", "Storgatan 7", "Storgatan 9"],
            city="Stockholm"
        )
        
        # Result includes:
        # - geometry: Clipped polygon (only right side)
        # - is_estimated_boundary: True
        # - boundary_estimation_note: "Building cut at 45% from left..."
    """
    resolver = FootprintResolver()
    footprints = resolver.resolve_with_smart_boundary(
        lat, lon, brf_addresses, city=city
    )
    
    if footprints:
        return footprints[0].to_dict()
    return {}

