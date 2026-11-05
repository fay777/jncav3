#!/usr/bin/env python3
"""Create frozen link-heterogeneity physical-network variants.

The script preserves topology, every node attribute, and every edge attribute
except ``bw``, ``max_bw``, and/or ``security``.  It is intended for zero-shot
robustness evaluation of already selected checkpoints, not for retraining.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.physical_network import PhysicalNetwork


VARIANTS = {
    # The bandwidth mixture retains the original expected mean (75) while
    # increasing dispersion: 60% Uniform[40,55], 40% Uniform[110,125].
    'bw_hetero': {'bandwidth': True, 'security': False},
    # Values 1 and 3 have equal counts (mean approximately 2), matching the
    # original Uniform{1,2,3} mean while removing the middle level.
    'security_hetero': {'bandwidth': False, 'security': True},
    'joint_hetero': {'bandwidth': True, 'security': True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def edge_key(edge):
    u, v = edge
    return tuple(sorted((str(u), str(v))))


def balanced_security_levels(count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.array([1] * ((count + 1) // 2) + [3] * (count // 2), dtype=int)
    rng.shuffle(values)
    return values


def heterogeneous_bandwidths(count: int, rng: np.random.Generator) -> np.ndarray:
    low_count = int(round(count * 0.60))
    values = np.empty(count, dtype=int)
    values[:low_count] = rng.integers(40, 56, size=low_count)
    values[low_count:] = rng.integers(110, 126, size=count - low_count)
    rng.shuffle(values)
    return values


def validate(source, variant, source_nodes, source_edges):
    if set(source.nodes) != set(source_nodes):
        raise AssertionError('Node set changed.')
    if {edge_key(edge) for edge in source.edges} != set(source_edges):
        raise AssertionError('Topology changed.')
    for node, attrs in source_nodes.items():
        if dict(source.nodes[node]) != attrs:
            raise AssertionError(f'Node attributes changed for node {node}.')
    for key, attrs in source_edges.items():
        u, v = next(edge for edge in source.edges if edge_key(edge) == key)
        changed = {'bw', 'max_bw', 'security'}
        current = dict(source.edges[u, v])
        if {name: value for name, value in current.items() if name not in changed} != {
            name: value for name, value in attrs.items() if name not in changed
        }:
            raise AssertionError(f'Unexpected edge attribute change on {key}.')
    if variant['bandwidth']:
        if any(source.edges[edge]['bw'] != source.edges[edge]['max_bw'] for edge in source.edges):
            raise AssertionError('max_bw must equal the reassigned bw.')
    if variant['security']:
        if not {source.edges[edge]['security'] for edge in source.edges}.issubset({1, 3}):
            raise AssertionError('Security variant must use only levels 1 and 3.')


def build_variant(source_dir: Path, output_root: Path, name: str, seed: int):
    variant = VARIANTS[name]
    output_dir = output_root / name / 'p_net'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing variant: {output_dir}')

    p_net = PhysicalNetwork.load_dataset(str(source_dir))
    source_nodes = {node: dict(p_net.nodes[node]) for node in p_net.nodes}
    source_edges = {edge_key(edge): dict(p_net.edges[edge]) for edge in p_net.edges}
    edges = sorted(p_net.edges, key=edge_key)
    rng = np.random.default_rng(seed)

    if variant['bandwidth']:
        for edge, bw in zip(edges, heterogeneous_bandwidths(len(edges), rng)):
            p_net.edges[edge]['bw'] = int(bw)
            p_net.edges[edge]['max_bw'] = int(bw)
    if variant['security']:
        for edge, level in zip(edges, balanced_security_levels(len(edges), rng)):
            p_net.edges[edge]['security'] = int(level)

    validate(p_net, variant, source_nodes, source_edges)
    output_dir.mkdir(parents=True, exist_ok=False)
    p_net.save_dataset(str(output_dir))

    values = lambda attribute: [p_net.edges[edge][attribute] for edge in edges]
    manifest = {
        'purpose': 'frozen zero-shot link-heterogeneity sensitivity evaluation',
        'source_p_net': str((source_dir / 'p_net.gml').resolve()),
        'source_p_net_sha256': sha256(source_dir / 'p_net.gml'),
        'variant': name,
        'seed': seed,
        'topology_changed': False,
        'node_attributes_changed': False,
        'changed_link_attributes': [
            attribute for attribute, enabled in (('bw', variant['bandwidth']), ('max_bw', variant['bandwidth']), ('security', variant['security']))
            if enabled
        ],
        'edge_count': len(edges),
        'bw': {'min': int(min(values('bw'))), 'max': int(max(values('bw'))), 'mean': float(np.mean(values('bw')))},
        'security': {'counts': {str(level): int(values('security').count(level)) for level in sorted(set(values('security')))},
                     'mean': float(np.mean(values('security')))},
        'output_p_net_sha256': sha256(output_dir / 'p_net.gml'),
    }
    (output_dir.parent / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Created {name}: {output_dir}')
    print(json.dumps(manifest, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-p-net', required=True, type=Path,
                        help='Directory containing the original p_net.gml')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--variants', nargs='+', choices=sorted(VARIANTS),
                        default=['bw_hetero', 'security_hetero', 'joint_hetero'])
    args = parser.parse_args()
    if not (args.source_p_net / 'p_net.gml').is_file():
        raise FileNotFoundError(f'Missing source p_net.gml: {args.source_p_net}')
    for index, name in enumerate(args.variants):
        build_variant(args.source_p_net, args.output_root, name, args.seed + index)


if __name__ == '__main__':
    main()
