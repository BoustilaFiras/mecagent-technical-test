"""
STL mesh sampling and Chamfer distance calculation using KD-Tree
Dependencies: trimesh, numpy, scipy

This module provides utilities for:
- Converting STL meshes to point clouds
- Computing Chamfer distance between 3D shapes
- Evaluating geometric similarity for CadQuery code generation
"""
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

def stl_to_points(stl_bytes: bytes, n_pts: int = 4096):
    """
    Convert STL bytes to point cloud
    
    Args:
        stl_bytes: STL file content as bytes
        n_pts: Number of points to sample from surface
    
    Returns:
        np.ndarray: Point cloud of shape (n_pts, 3)
    """
    mesh = trimesh.load_mesh(trimesh.util.wrap_as_stream(stl_bytes), file_type="stl")
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts)
    return pts.astype("float32")

def chamfer_kdtree(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Chamfer L2 distance between two point clouds using KD-Tree
    
    Args:
        p: First point cloud (N, 3)
        q: Second point cloud (M, 3)
    
    Returns:
        float: Normalized Chamfer distance [0, 1] (0 = identical)
        
    Note:
        Chamfer distance is symmetric: d(p,q) = d(q,p)
        It measures the average distance from each point to its nearest neighbor
    """
    # Build KD-trees for efficient nearest neighbor search
    tree_p, tree_q = KDTree(p), KDTree(q)
    
    # Calculate distances from q to nearest points in p
    d1 = tree_p.query(q, k=1)[0].mean()
    
    # Calculate distances from p to nearest points in q
    d2 = tree_q.query(p, k=1)[0].mean()
    
    # Return symmetric Chamfer distance
    return (d1 + d2) / 2.0
    d2 = tree_q.query(p, k=1)[0].mean()
    
    # Return symmetric Chamfer distance
    return (d1 + d2) / 2.0
