#!/usr/bin/env python3
"""
Generate bbox.usda file from labels.json.

This script reads bounding box information from labels.json and generates
a USD file with BasisCurves representing the bounding boxes.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Try to import USD Python bindings, but fall back to text-based generation
USE_USD_API = False
try:
    from pxr import Gf, Sdf, Usd, UsdGeom
    USE_USD_API = True
except ImportError:
    pass


def transform_coordinate(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Transform coordinates from labels.json format to USD format.
    
    Based on analysis of existing bbox.usda files:
    - USD x = -labels z
    - USD y = labels x  
    - USD z = -labels y
    
    This transforms from the coordinate system used in labels.json
    to USD's Z-up coordinate system.
    
    Args:
        x: X coordinate from labels.json
        y: Y coordinate from labels.json
        z: Z coordinate from labels.json
    
    Returns:
        Tuple of (usd_x, usd_y, usd_z)
    """
    return (-z, x, -y)


def generate_bbox_edges(corners: List[dict]) -> List[Tuple[float, float, float]]:
    """
    Generate 12 edges (24 points) for a bounding box from 8 corner points.
    
    The 8 corners are expected to be in standard order:
    - corners[0-3]: one face (front face after coordinate transform)
    - corners[4-7]: opposite face (back face after coordinate transform)
    
    After coordinate transformation:
    - corners[0-3]: front face (x = x_min in USD)
    - corners[4-7]: back face (x = x_max in USD)
    
    Edge order matches bbox.usda pattern:
    - Edges 0-3: front face (0->1->2->3->0)
    - Edges 4-7: back face (4->5->6->7->4)
    - Edges 8-11: vertical edges (0->4, 1->5, 2->6, 3->7)
    
    Args:
        corners: List of 8 corner dicts, each with 'x', 'y', 'z' keys
    
    Returns:
        List of 24 points representing 12 edges (each edge has 2 points)
    """
    if len(corners) != 8:
        raise ValueError(f"Expected 8 corners, got {len(corners)}")
    
    # Convert corners to tuples and transform coordinates
    pts = []
    for corner in corners:
        usd_pt = transform_coordinate(corner['x'], corner['y'], corner['z'])
        pts.append(usd_pt)
    
    # Define edges as pairs of corner indices
    # 12 edges matching the exact pattern from bbox.usda:
    edges = [
        # Front face edges (corners 0-3)
        (pts[0], pts[1]),  # edge 0: 0->1
        (pts[1], pts[2]),  # edge 1: 1->2
        (pts[2], pts[3]),  # edge 2: 2->3
        (pts[3], pts[0]),  # edge 3: 3->0
        
        # Back face edges (corners 4-7)
        (pts[4], pts[5]),  # edge 4: 4->5
        (pts[5], pts[6]),  # edge 5: 5->6
        (pts[6], pts[7]),  # edge 6: 6->7
        (pts[7], pts[4]),  # edge 7: 7->4
        
        # Vertical edges connecting front to back
        (pts[0], pts[4]),  # edge 8: 0->4
        (pts[1], pts[5]),  # edge 9: 1->5
        (pts[2], pts[6]),  # edge 10: 2->6
        (pts[3], pts[7]),  # edge 11: 3->7
    ]
    
    # Flatten edges into list of points (each edge contributes 2 points)
    points = []
    for edge in edges:
        points.extend(edge)
    
    return points


def format_point(pt: Tuple[float, float, float]) -> str:
    """Format a point tuple as a USD point3f value."""
    return f"({pt[0]}, {pt[1]}, {pt[2]})"


def generate_bbox_usda_text(labels_json_path: Path, output_usda_path: Path) -> None:
    """
    Generate bbox.usda file from labels.json using text-based generation.
    
    This version doesn't require USD Python bindings and generates the file directly.
    
    Args:
        labels_json_path: Path to input labels.json file
        output_usda_path: Path to output bbox.usda file
    """
    # Load labels.json
    with open(labels_json_path, 'r') as f:
        labels = json.load(f)
    
    # Generate USD file content
    lines = [
        "#usda 1.0",
        "(",
        "    defaultPrim = \"World\"",
        "    metersPerUnit = 1",
        "    upAxis = \"Z\"",
        ")",
        "",
        "def Xform \"World\"",
        "{",
    ]
    
    # Create BasisCurves for each bounding box
    for label_data in labels:
        ins_id = label_data['ins_id']
        label_name = label_data['label']
        bbox_name = f"BBox_{ins_id}_{label_name}"
        
        # Generate edge points
        edge_points = generate_bbox_edges(label_data['bounding_box'])
        
        # Format points as a single line
        points_str = ", ".join(format_point(pt) for pt in edge_points)
        
        # Generate BasisCurves definition
        lines.append(f'    def BasisCurves "{bbox_name}"')
        lines.append("    {")
        lines.append("        int[] curveVertexCounts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]")
        lines.append(f"        point3f[] points = [{points_str}]")
        lines.append("        color3f[] primvars:displayColor = [(1, 0, 0)]")
        lines.append("        uniform token type = \"linear\"")
        lines.append("        float[] widths = [0.01]")
        lines.append("    }")
        lines.append("")
    
    # Close World Xform
    lines.append("}")
    
    # Write to file
    with open(output_usda_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Successfully generated {output_usda_path}")
    print(f"Created {len(labels)} bounding boxes")


def generate_bbox_usda_api(labels_json_path: Path, output_usda_path: Path) -> None:
    """
    Generate bbox.usda file from labels.json using USD Python API.
    
    Args:
        labels_json_path: Path to input labels.json file
        output_usda_path: Path to output bbox.usda file
    """
    # Load labels.json
    with open(labels_json_path, 'r') as f:
        labels = json.load(f)
    
    # Create USD stage
    stage = Usd.Stage.CreateNew(str(output_usda_path))
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")
    
    # Define World Xform
    world_path = "/World"
    world_xform = UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", "World")
    
    # Create BasisCurves for each bounding box
    for label_data in labels:
        ins_id = label_data['ins_id']
        label_name = label_data['label']
        bbox_name = f"BBox_{ins_id}_{label_name}"
        
        # Generate edge points
        edge_points = generate_bbox_edges(label_data['bounding_box'])
        
        # Create BasisCurves prim
        curves_path = f"{world_path}/{bbox_name}"
        curves = UsdGeom.BasisCurves.Define(stage, curves_path)
        
        # Set curve type to linear
        curves.GetTypeAttr().Set("linear")
        
        # Set curve vertex counts: 12 edges, each with 2 vertices
        curves.GetCurveVertexCountsAttr().Set([2] * 12)
        
        # Set points
        points_array = Gf.Vec3fArray([Gf.Vec3f(pt) for pt in edge_points])
        curves.GetPointsAttr().Set(points_array)
        
        # Set display color (red)
        color_primvar = UsdGeom.Primvar(curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform))
        color_primvar.Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        
        # Set width
        widths_attr = curves.CreateWidthsAttr()
        widths_attr.Set([0.01])
    
    # Save stage
    stage.GetRootLayer().Save()
    print(f"Successfully generated {output_usda_path}")
    print(f"Created {len(labels)} bounding boxes")


def generate_bbox_usda(labels_json_path: Path, output_usda_path: Path) -> None:
    """
    Generate bbox.usda file from labels.json.
    
    Uses USD Python API if available, otherwise falls back to text-based generation.
    
    Args:
        labels_json_path: Path to input labels.json file
        output_usda_path: Path to output bbox.usda file
    """
    if USE_USD_API:
        generate_bbox_usda_api(labels_json_path, output_usda_path)
    else:
        generate_bbox_usda_text(labels_json_path, output_usda_path)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        labels_json_path = Path(sys.argv[1])
    else:
        # Default: look for labels.json in interio directory
        script_dir = Path(__file__).parent
        labels_json_path = script_dir / "interio" / "labels.json"
    
    if len(sys.argv) > 2:
        output_usda_path = Path(sys.argv[2])
    else:
        # Default: output to bbox.usda in same directory as labels.json
        output_usda_path = labels_json_path.parent / "bbox.usda"
    
    if not labels_json_path.exists():
        print(f"Error: labels.json not found at {labels_json_path}")
        sys.exit(1)
    
    generate_bbox_usda(labels_json_path, output_usda_path)


if __name__ == "__main__":
    main()

