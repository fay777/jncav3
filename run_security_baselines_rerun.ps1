# Re-run the two differentiated security-aware adapted baselines on the five
# frozen JNCA test workloads.  Corrected results are co-located with the
# formal CPU baselines.  Replacing the two known-invalid legacy result trees
# requires explicit opt-in and never affects other baseline methods.

param(
    [switch]$ReplaceInvalidResults
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$resultRoot = 'runs_formal_20261022/baselines_cpu'
$logRoot = Join-Path $resultRoot 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$solvers = @('appf_dsi_adapted', 'ssd_ose_adapted')
$seeds = 20001..20005

foreach ($solver in $solvers) {
    $legacySolverDir = Join-Path $resultRoot $solver
    if (Test-Path $legacySolverDir) {
        if (-not $ReplaceInvalidResults) {
            throw "Existing $solver results are known-invalid. Re-run with -ReplaceInvalidResults to replace only this method's legacy result tree."
        }
        Remove-Item -LiteralPath $legacySolverDir -Recurse -Force
    }
}

foreach ($solver in $solvers) {
    foreach ($seed in $seeds) {
        $runDir = "$resultRoot/$solver/test_seed$seed"
        $stdout = Join-Path $logRoot "$solver-$seed.out.log"
        $stderr = Join-Path $logRoot "$solver-$seed.err.log"

        Write-Host "[$(Get-Date -Format s)] $solver, test seed $seed"
        # tqdm writes normal progress updates to stderr.  Do not let
        # ErrorActionPreference=Stop turn those updates into a terminating
        # NativeCommandError; decide success solely from Python's exit code.
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & python main.py `
                --solver_name $solver `
                --p_net_dataset_path benchmark_jnca_v1/p_net `
                --v_nets_dataset_path "benchmark_jnca_v1/test/seed_$seed" `
                --seed $seed `
                --shortest_method bfs_shortest `
                --k_shortest 1 `
                --use_cuda false `
                --if_save_records true `
                --if_temp_save_records false `
                --run_dir $runDir 1> $stdout 2> $stderr
            $pythonExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($pythonExitCode -ne 0) {
            throw "$solver on seed $seed failed. Inspect $stderr"
        }
        if (-not (Test-Path (Join-Path $runDir 'global_summary.csv'))) {
            throw "$solver on seed $seed did not write global_summary.csv"
        }
    }
}

Write-Host "Completed. Results are in $resultRoot"
