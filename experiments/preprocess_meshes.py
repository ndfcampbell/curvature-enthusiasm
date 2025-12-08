"""Preprocessing script for mesh alignment and scaling.

This script prepares meshes for training by:
1. Loading raw meshes from dataset directories
2. Scaling to desired volume
3. Aligning source and target meshes
4. Saving preprocessed meshes

Usage:
    python preprocess_meshes.py --dataset MANO --source 01_01r --target 01_02r
    python preprocess_meshes.py --dataset DFAUST --source 50002_hips --target 50002_jiggle_on_toes
    python preprocess_meshes.py --dataset TOSCA --source cat0 --target cat1 --volume 4.18879
"""
import argparse
from curvature_enthusiasm.utils.preprocessing_utils import preprocess_mesh_pair


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess meshes for training")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., MANO, DFAUST, TOSCA)")
    parser.add_argument("--source", type=str, required=True,
                        help="Source pose/mesh name (without extension)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target pose/mesh name (without extension)")
    # parser.add_argument("--volume", type=float, default=None,
    #                     help="Desired volume (default: 4π/3 ≈ 4.18879)")
    # parser.add_argument("--input_dir", type=str, default=None,
    #                     help="Input directory (default: data/{dataset}/meshes)")
    # parser.add_argument("--output_dir", type=str, default=None,
    #                     help="Output directory (default: data/{dataset}/preprocessed)")
    # parser.add_argument("--align_method", type=str, default='centroid',
    #                     choices=['centroid', 'procrustes', 'none'],
    #                     help="Alignment method: 'centroid' (translate only), "
    #                          "'procrustes' (rigid transform), 'none'")
    # parser.add_argument("--scale_method", type=str, default='volume',
    #                     choices=['volume', 'independent', 'none'],
    #                     help="Scaling method: 'volume' (both to same volume), "
    #                          "'independent' (each to target volume), 'none'")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    print("=" * 60)
    print("MESH PREPROCESSING")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Source pose: {args.source}")
    print(f"Target pose: {args.target}")
    print(f"Alignment method: {args.align_method}")
    print(f"Scale method: {args.scale_method}")
    if args.volume:
        print(f"Target volume: {args.volume:.6f}")
    print("")

    preprocess_mesh_pair(
        dataset_name=args.dataset,
        source_pose=args.source,
        target_pose=args.target,
        desired_volume=args.volume,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        align_method=args.align_method,
        scale_method=args.scale_method,
        save_to_preprocessed=True
    )

    print("\nPreprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()