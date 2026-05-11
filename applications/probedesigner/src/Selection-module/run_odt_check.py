#!/usr/bin/env python
"""
CLI wrapper for ODT probe designability check (filter_only mode).

Wraps CompleteProbeDesignPipeline from _design_probes.py and writes
per-gene results in a format consumed by the App's run_odt_check() runner.

Exit codes:
    0  - success
    1  - pipeline error
    2  - oligo_designer_toolsuite not installed

Usage:
    python run_odt_check.py --genes genes.txt --output ./odt_out --pipeline scrinshot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure this module directory is importable
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from _design_probes import CompleteProbeDesignPipeline, ODT_AVAILABLE


def read_gene_list(path: str) -> list:
    """Read gene list from a file (one gene per line, # lines ignored)."""
    genes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            genes.append(line)
    return genes


def main() -> int:
    # Check package availability before parsing args so the error is immediate
    if not ODT_AVAILABLE:
        print("ERROR: oligo_designer_toolsuite is not installed.", file=sys.stderr)
        print("Install it with: pip install oligo-designer-toolsuite", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        description='ODT probe designability check (filter_only mode)'
    )
    parser.add_argument('--genes', required=True,
                        help='Path to file with gene names (one per line)')
    parser.add_argument('--output', required=True,
                        help='Output directory for results')
    parser.add_argument('--pipeline', default='scrinshot',
                        choices=['scrinshot', 'merfish', 'seqfishplus'],
                        help='Probe design pipeline (default: scrinshot)')
    parser.add_argument('--species', default='mus_musculus',
                        help='Species (default: mus_musculus)')
    parser.add_argument('--annotation-source', default='ensembl',
                        choices=['ensembl', 'ncbi'],
                        help='Annotation source (default: ensembl)')
    parser.add_argument('--annotation-release', default='110',
                        help='Annotation release version (default: 110)')
    parser.add_argument('--taxon', default='vertebrate_mammalian',
                        help='NCBI taxon (default: vertebrate_mammalian)')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Parallel jobs (default: 4)')
    parser.add_argument('--gtf-file', default=None,
                        help='Path to existing GTF file (skip download)')
    parser.add_argument('--genome-file', default=None,
                        help='Path to existing genome FASTA (skip download)')
    parser.add_argument('--probe-length-min', type=int, default=None,
                        help='Minimum probe length')
    parser.add_argument('--probe-length-max', type=int, default=None,
                        help='Maximum probe length')
    parser.add_argument('--n-sets', type=int, default=None,
                        help='Number of probe sets')
    parser.add_argument('--set-size-min', type=int, default=None,
                        help='Minimum probes per set')
    parser.add_argument('--set-size-opt', type=int, default=None,
                        help='Optimal probes per set')

    args = parser.parse_args()

    gene_list = read_gene_list(args.genes)
    print(f"Running ODT designability check on {len(gene_list)} genes...")
    print(f"  Pipeline : {args.pipeline.upper()}")
    print(f"  Species  : {args.species}")
    print(f"  Output   : {args.output}")

    pipeline = CompleteProbeDesignPipeline(
        pipeline_type=args.pipeline,
        output_dir=args.output,
        species=args.species,
        annotation_source=args.annotation_source,
        annotation_release=args.annotation_release,
        taxon=args.taxon,
        design_mode='filter_only',
        n_jobs=args.n_jobs,
    )

    # Inject pre-existing reference files if provided
    if args.gtf_file:
        pipeline.gtf_file = args.gtf_file
        print(f"  Using GTF    : {args.gtf_file}")
    if args.genome_file:
        pipeline.genome_file = args.genome_file
        print(f"  Using genome : {args.genome_file}")

    # Build optional probe design kwargs
    design_kwargs: dict = {}
    if args.probe_length_min is not None:
        design_kwargs['target_probe_length_min'] = args.probe_length_min
    if args.probe_length_max is not None:
        design_kwargs['target_probe_length_max'] = args.probe_length_max
    if args.n_sets is not None:
        design_kwargs['n_sets'] = args.n_sets
    if args.set_size_min is not None:
        design_kwargs['set_size_min'] = args.set_size_min
    if args.set_size_opt is not None:
        design_kwargs['set_size_opt'] = args.set_size_opt

    try:
        results = pipeline.run_complete_pipeline(gene_list=gene_list, **design_kwargs)
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Write outputs consumed by App/_runner.py::run_odt_check()
    results_dir = Path(args.output) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'n_requested': results['n_requested'],
        'n_successful': results['n_successful'],
        'successful_gene_symbols': results['successful_gene_symbols'],
        'pipeline': args.pipeline,
        'design_mode': 'filter_only',
    }
    (results_dir / 'probe_design_summary.json').write_text(
        json.dumps(summary, indent=2)
    )

    with open(results_dir / 'successful_genes.txt', 'w') as f:
        f.write('# Successfully designed genes\n')
        for gene in results['successful_gene_symbols']:
            f.write(f'{gene}\n')

    print(
        f"\n✓ ODT check complete. "
        f"{results['n_successful']}/{results['n_requested']} genes pass ODT filter."
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
