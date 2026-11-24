# Copyright 2023 DeepMind Technologies Limited. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for creating icosahedral meshes."""

# Standard library
import itertools
from typing import List, NamedTuple, Sequence, Tuple

# Third-party
import numpy as np
from scipy.spatial import transform
import tqdm

# Optional acceleration libraries
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try: 
    import dask.array as da
    HAS_DASK = True
except ImportError: 
    HAS_DASK = False
    da = None


class TriangularMesh(NamedTuple):
    """Data structure for triangular meshes. 

    Attributes:
      vertices: spatial positions of the vertices of the mesh of shape
          [num_vertices, num_dims].
      faces: triangular faces of the mesh of shape [num_faces, 3].  Contains
          integer indices into `vertices`.

    """

    vertices: np. ndarray
    faces: np. ndarray


def merge_meshes(mesh_list:  Sequence[TriangularMesh]) -> TriangularMesh:
    """Merges all meshes into one.  Assumes the last mesh is the finest.

    Args:
       mesh_list:  Sequence of meshes, from coarse to fine refinement levels.  The
         vertices and faces may contain those from preceding, coarser levels.

    Returns:
       `TriangularMesh` for which the vertices correspond to the highest
       resolution mesh in the hierarchy, and the faces are the join set of the
       faces at all levels of the hierarchy. 
    """
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(
            mesh_i. vertices, mesh_ip1.vertices[: num_nodes_mesh_i]
        )

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh. faces for mesh in mesh_list], axis=0),
    )


def get_hierarchy_of_triangular_meshes_for_sphere(
    splits: int,
    use_cupy: bool = False,
    use_dask: bool = True,
) -> List[TriangularMesh]:
    """Returns a sequence of meshes, each with triangularization sphere.

    Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
    circumscribed unit sphere.  Then, each triangular face is iteratively
    subdivided into 4 triangular faces `splits` times. The new vertices are then
    projected back onto the unit sphere.  All resulting meshes are returned in a
    list, from lowest to highest resolution.

    The vertices in each face are specified in counter-clockwise order as
    observed from the outside the icosahedron.

    Args:
       splits: How many times to split each triangle. 
       use_cupy: Use CuPy for GPU acceleration (requires cupy installed).
       use_dask: Use Dask for parallel computation (requires dask installed).
    Returns:
       Sequence of `TriangularMesh`s of length `splits + 1` each with: 

         vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
         faces: [num_faces, 3] with triangular faces joining sets of 3 vertices. 
             Each row contains three indices into the vertices array, indicating
             the vertices adjacent to the face.  Always with positive orientation
             (counterclock-wise when looking from the outside).
    """
    current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in tqdm.trange(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(
            current_mesh, use_cupy=use_cupy, use_dask=use_dask
        )
        output_meshes.append(current_mesh)
    return output_meshes


def get_icosahedron() -> TriangularMesh: 
    """Returns a regular icosahedral mesh with circumscribed unit sphere. 

    See https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
    for details on the construction of the regular icosahedron.

    The vertices in each face are specified in counter-clockwise order as observed
    from the outside of the icosahedron.

    Returns:
       TriangularMesh with: 

       vertices: [num_vertices=12, 3] vertex positions in 3D, all with unit norm.
       faces: [num_faces=20, 3] with triangular faces joining sets of 3 vertices. 
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation (
           counterclock-wise when looking from the outside).

    """
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1.0, -1.0]: 
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.0))
            vertices.append((0.0, c1, c2))
            vertices. append((c2, 0.0, c1))

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    # I did this manually, checking the orientation one by one. 
    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]

    # By default the top is an aris parallel to the Y axis.
    # Need to rotate around the y axis by half the supplementary to the
    # angle between faces divided by two to get the desired orientation.
    #                          /O\  (top arist)
    #                     /          \                           Z
    # (adjacent face)/                    \  (adjacent face)     ^
    #           /     angle_between_faces      \                 |
    #      /                                        \            |
    #  /                                                 \      YO-----> X
    # This results in: 
    #  (adjacent faceis now top plane)
    #  ----------------------O\  (top arist)
    #                           \
    #                             \
    #                               \     (adjacent face)
    #                                 \
    #                                   \
    #                                     \

    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform. Rotation. from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

    return TriangularMesh(
        vertices=vertices. astype(np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh,
    use_cupy: bool = False,
    use_dask: bool = False,
) -> TriangularMesh:
    """Splits each triangular face into 4 triangles keeping the orientation. 
    
    Vectorized implementation that avoids loops for better performance.
    
    Args:
        triangular_mesh: Input mesh to split. 
        use_cupy: Use CuPy for GPU acceleration (requires cupy installed).
        use_dask: Use Dask for parallel computation (requires dask installed).
    
    Returns:
        New TriangularMesh with each face split into 4 triangles.
    """
    # Choose array library based on parameters
    if use_dask and HAS_DASK:
        xp = da
        import ipdb; ipdb.set_trace()
        vertices = da.from_array(triangular_mesh.vertices, chunks='auto').map_blocks(cp.asarray)
        faces = da.from_array(triangular_mesh.faces, chunks='auto').map_blocks(cp.asarray)
    else:
        xp = np
        vertices = triangular_mesh.vertices
        faces = triangular_mesh.faces

    num_parent_vertices = vertices.shape[0]
    
    # Extract vertex indices for each face
    ind1 = faces[:, 0]
    ind2 = faces[: , 1]
    ind3 = faces[:, 2]
    
    # Create edge pairs and sort them to create unique identifiers
    # Each edge appears twice (once per adjacent face), but with consistent ordering
    edges = xp.concatenate([
        xp.stack([xp.minimum(ind1, ind2), xp.maximum(ind1, ind2)], axis=1),
        xp.stack([xp.minimum(ind2, ind3), xp.maximum(ind2, ind3)], axis=1),
        xp.stack([xp. minimum(ind3, ind1), xp.maximum(ind3, ind1)], axis=1),
    ], axis=0)
    
    # Find unique edges and inverse mapping
    if use_dask and HAS_DASK: 
        # For Dask, compute to NumPy for unique operation, then convert back
        edges_np = edges
        edges_np = da.asarray(edges_np, dtype=da.int64)
        base = da.int64(num_parent_vertices) + 1
        edges_view = edges_np[:, 0] + edges_np[: , 1] * base
        unique_edges_view, inverse_indices = da.unique(edges_view, return_inverse=True)
        unique_edge_v0 = unique_edges_view % (num_parent_vertices + 1)
        unique_edge_v1 = unique_edges_view // (num_parent_vertices + 1)
        unique_edges = da.from_array(da.stack([unique_edge_v0, unique_edge_v1], axis=1))
        inverse_indices = da.from_array(inverse_indices)
    else:
        # NumPy version
        edges_int64 = edges.astype(np.int64)
        base = np.int64(num_parent_vertices) + 1
        edges_view = edges_int64[:, 0] + edges_int64[:, 1] * base
        unique_edges_view, inverse_indices = np.unique(edges_view, return_inverse=True)
        unique_edge_v0 = unique_edges_view % (num_parent_vertices + 1)
        unique_edge_v1 = unique_edges_view // (num_parent_vertices + 1)
        unique_edges = np.stack([unique_edge_v0, unique_edge_v1], axis=1)
    
    # Compute midpoints for unique edges
    edge_midpoints = (vertices[unique_edges[: , 0]] + 
                         vertices[unique_edges[:, 1]]) / 2.0
    
    # Project to unit sphere
    norms = xp.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    edge_midpoints = edge_midpoints / norms
    
    # New vertices are old vertices + edge midpoints
    all_vertices = xp.concatenate([vertices, edge_midpoints], axis=0)
    
    # Map edge indices to new vertex indices
    num_faces = faces.shape[0]
    new_vertex_indices = inverse_indices + num_parent_vertices
    
    # Split into the three edge midpoint indices per face
    ind12 = new_vertex_indices[:num_faces]
    ind23 = new_vertex_indices[num_faces:2*num_faces]
    ind31 = new_vertex_indices[2*num_faces:]
    
    # Create 4 new faces for each original face
    new_faces = xp.stack([
        xp.stack([ind1, ind12, ind31], axis=1),  # Triangle 1
        xp.stack([ind12, ind2, ind23], axis=1),  # Triangle 2
        xp.stack([ind31, ind23, ind3], axis=1),  # Triangle 3
        xp.stack([ind12, ind23, ind31], axis=1), # Triangle 4 (center)
    ], axis=1).reshape(-1, 3)
    
    return TriangularMesh(
        vertices=all_vertices.astype(np.float32),
        faces=new_faces.astype(np.int32),
    )


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms polygonal faces to sender and receiver indices.

    It does so by transforming every face into N_i edges.  Such if the triangular
    face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

    If all faces have consistent orientation, and the surface represented by the
    faces is closed, then every edge in a polygon with a certain orientation
    is also part of another polygon with the opposite orientation.  In this
    situation, the edges returned by the method are always bidirectional.

    Args:
      faces: Integer array of shape [num_faces, 3].  Contains node indices
          adjacent to each face. 
    Returns:
      Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3]. 

    """
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def get_last_triangular_mesh_for_sphere(
    splits: int,
    use_cupy:  bool = False,
    use_dask: bool = False,
) -> TriangularMesh:
    return get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits, use_cupy=use_cupy, use_dask=use_dask
    )[-1]
