"""Create an immutable JNCA train/validation/test benchmark package."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Permit ``python tools/create_benchmark.py`` from the repository root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.physical_network import PhysicalNetwork
from data.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from utils import read_setting


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hashes(root):
    return {
        os.path.relpath(os.path.join(directory, filename), root).replace(os.sep, '/'): sha256(os.path.join(directory, filename))
        for directory, _, filenames in os.walk(root)
        for filename in sorted(filenames)
    }


def require_new(path):
    if os.path.exists(path):
        raise FileExistsError(f'Refusing to overwrite existing benchmark split: {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-dir', default='benchmark_jnca_v1')
    parser.add_argument('--p-net-source', required=True, help='Existing directory containing the fixed p_net.gml')
    parser.add_argument('--v-sim-setting', default='settings/v_sim_setting.yaml')
    parser.add_argument('--validation-seed', type=int, default=10001)
    parser.add_argument('--test-seeds', type=int, nargs='+', default=[20001, 20002, 20003, 20004, 20005])
    args = parser.parse_args()

    root = os.path.abspath(args.benchmark_dir)
    if os.path.exists(root):
        raise FileExistsError(f'Refusing to overwrite benchmark directory: {root}')
    if not os.path.isfile(os.path.join(args.p_net_source, 'p_net.gml')):
        raise FileNotFoundError(f'No p_net.gml in {args.p_net_source}')
    os.makedirs(root)

    p_net = PhysicalNetwork.load_dataset(args.p_net_source)
    p_net_dir = os.path.join(root, 'p_net')
    os.makedirs(p_net_dir)
    p_net.save_dataset(p_net_dir)

    setting = read_setting(args.v_sim_setting)
    split_seeds = {'validation/seed_' + str(args.validation_seed): args.validation_seed}
    split_seeds.update({'test/seed_' + str(seed): seed for seed in args.test_seeds})
    for relative_dir, seed in split_seeds.items():
        destination = os.path.join(root, relative_dir)
        require_new(destination)
        simulator = VirtualNetworkRequestSimulator.from_setting(setting)
        simulator.renew(v_nets=True, events=True, seed=seed)
        simulator.save_dataset(destination)

    with open(os.path.join(root, 'README.md'), 'w', encoding='utf-8') as handle:
        handle.write('# Frozen JNCA benchmark\n\nThis package is immutable. Validation and test runs must load only the split directories recorded in manifest.json.\n')
    manifest = {
        'schema': 'benchmark_jnca_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_revision': git_revision(),
        'p_net_source': os.path.abspath(args.p_net_source),
        'v_sim_setting': os.path.abspath(args.v_sim_setting),
        'validation_seed': args.validation_seed,
        'test_seeds': args.test_seeds,
        'sha256': collect_hashes(root),
    }
    with open(os.path.join(root, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'Created frozen benchmark: {root}')


if __name__ == '__main__':
    main()
