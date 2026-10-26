#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from checkpoint_dag_diffusion.control import select_scaffold_checkpoint
from checkpoint_dag_diffusion.global_sampling import sample_global_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample global checkpoint diffusion.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--global_graph", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--start_code", type=str, default=None)
    parser.add_argument("--start_level", type=int, default=None)
    parser.add_argument("--scaffold_smiles", type=str, default=None)
    parser.add_argument("--scaffold_codebook", type=str, default=None)
    parser.add_argument("--scaffold_level", type=int, default=None)
    parser.add_argument("--scaffold_mode", type=str, default=None)
    parser.add_argument("--scaffold_exact_quotient", action="store_true")
    parser.add_argument("--ignore_scaffold_boundary", action="store_true")
    parser.add_argument("--waypoint_code", type=str, default=None)
    parser.add_argument("--waypoint_level", type=int, default=None)
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Comma-separated complete path, e.g. C42@L3,C137@L1,C301@L0.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=1.3)
    parser.add_argument("--transition_temperature", type=float, default=1.0)
    parser.add_argument("--denoise_temperature", type=float, default=0.9)
    parser.add_argument("--none_bias", type=float, default=2.0)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--max_path_steps", type=int, default=8)
    parser.add_argument("--max_refinement_retries", type=int, default=2)
    parser.add_argument(
        "--variant",
        choices=["ccd", "exact_plan_final_diffusion"],
        default="ccd",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    control_metadata = None
    if args.scaffold_smiles is not None:
        if args.start_code is not None:
            raise ValueError("Use either --start_code or --scaffold_smiles, not both.")
        if args.scaffold_codebook is None:
            raise ValueError("--scaffold_codebook is required with --scaffold_smiles.")
        selection = select_scaffold_checkpoint(
            scaffold_smiles=args.scaffold_smiles,
            codebook_path=args.scaffold_codebook,
            global_graph_path=args.global_graph,
            level=args.scaffold_level,
            mode=args.scaffold_mode,
            exact_quotient=args.scaffold_exact_quotient,
            require_boundary=not args.ignore_scaffold_boundary,
            device=args.device,
        )
        args.start_code = f"C{selection.selected_global_id}"
        args.start_level = selection.selected_level
        control_metadata = {"scaffold_selection": selection.to_dict()}
    summary = sample_global_pipeline(
        checkpoint=args.checkpoint,
        global_graph_path=args.global_graph,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        start_code=args.start_code,
        start_level=args.start_level,
        waypoint_code=args.waypoint_code,
        waypoint_level=args.waypoint_level,
        path=None if args.path is None else [value.strip() for value in args.path.split(",") if value.strip()],
        device=args.device,
        guidance=args.guidance,
        transition_temperature=args.transition_temperature,
        denoise_temperature=args.denoise_temperature,
        none_bias=args.none_bias,
        max_degree=args.max_degree,
        max_path_steps=args.max_path_steps,
        max_refinement_retries=args.max_refinement_retries,
        variant=args.variant,
        control_metadata=control_metadata,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
