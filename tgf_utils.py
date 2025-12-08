#!/usr/bin/env python3
"""
TGF (Trivial Graph Format) file I/O utilities.

TGF format:
    <vertex_id> <x> <y> <z>
    ...
    #
    <edge_start_id> <edge_end_id>
    ...
"""

import numpy as np


def read_tgf(filename):
    """
    Read a TGF skeleton file.

    Args:
        filename: Path to .tgf file

    Returns:
        C: (n, 3) array of joint positions
        BE: (m, 2) array of bone edges (indices into C)
    """
    vertices = []
    edges = []
    reading_edges = False
    vertex_id_map = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for separator
            if line == '#':
                reading_edges = True
                continue

            parts = line.split()

            if not reading_edges:
                # Reading vertices: <id> <x> <y> <z>
                if len(parts) >= 4:
                    vertex_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertex_id_map[vertex_id] = len(vertices)
                    vertices.append([x, y, z])
            else:
                # Reading edges: <start_id> <end_id>
                if len(parts) >= 2:
                    start_id = int(parts[0])
                    end_id = int(parts[1])
                    # Map original IDs to array indices
                    start_idx = vertex_id_map[start_id]
                    end_idx = vertex_id_map[end_id]
                    edges.append([start_idx, end_idx])

    C = np.array(vertices, dtype=np.float64)
    BE = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)

    return C, BE


def write_tgf(filename, C, BE):
    """
    Write a TGF skeleton file.

    Args:
        filename: Path to output .tgf file
        C: (n, 3) array of joint positions
        BE: (m, 2) array of bone edges (indices into C)
    """
    with open(filename, 'w') as f:
        # Write vertices
        for i, pos in enumerate(C):
            f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Write separator
        f.write("#\n")

        # Write edges
        for edge in BE:
            f.write(f"{edge[0]} {edge[1]}\n")


if __name__ == '__main__':
    # Test
    import sys

    if len(sys.argv) > 1:
        C, BE = read_tgf(sys.argv[1])
        print(f"Loaded {len(C)} vertices and {len(BE)} edges")
        print(f"Vertices:\n{C}")
        print(f"Edges:\n{BE}")