#!/bin/bash
# Final evaluation chain for the revision (run after all training has finished
# and after `evaluate.py --merge` has produced the 10-seed daily_costs.csv).
cd "$(dirname "$0")"
PY=${PYTHON:-python}
set -x
mkdir -p logs/rev generated
S10="0,1,2,3,4,5,6,7,8,9"

# 1. evaluations of the agents trained last (parallel)
$PY rev_eval_extra.py cfga   > logs/rev/eval_cfga.log 2>&1 &
$PY rev_eval_extra.py tariff > logs/rev/eval_tariff.log 2>&1 &
$PY scale_run.py eval 2020 0,1,2,3,4 > logs/rev/scale_eval_2020.log 2>&1 &
$PY scale_run.py eval 2023 0,1,2,3,4 > logs/rev/scale_eval_2023.log 2>&1 &

# 2. secondary analyses rerun with 10 seeds (parallel)
$PY explain_stats.py > logs/rev/explain_stats10.log 2>&1 &
$PY -c "import explain_audit as e; e.main(seeds=tuple(range(10)))" > logs/rev/explain_audit10.log 2>&1 &
$PY costmodel_rev.py > logs/rev/costmodel10.log 2>&1 &
$PY policy_agnostic.py > logs/rev/policy_agnostic10.log 2>&1 &
wait

# 3. forecast scenarios with 10 seeds (two streams)
rm -f results/forecast_rev_*.csv
( $PY forecast_rev.py $S10 real > logs/rev/fc10_real.log 2>&1
  $PY forecast_rev.py $S10 ar1_rho0.3 > logs/rev/fc10_r03.log 2>&1
  $PY forecast_rev.py $S10 ar1_rho0.7 > logs/rev/fc10_r07.log 2>&1 ) &
( $PY forecast_rev.py $S10 real+intraday > logs/rev/fc10_realint.log 2>&1
  $PY forecast_rev.py $S10 ar1_rho0.95 > logs/rev/fc10_r095.log 2>&1
  $PY forecast_rev.py $S10 ar1_rho0.7+intraday > logs/rev/fc10_r07int.log 2>&1 ) &
$PY -c "import robust_ext as r; r.main(seeds=tuple(range(10)))" > logs/rev/robust_ext10.log 2>&1 &
wait
$PY forecast_rev.py summary > logs/rev/fc10_summary.log 2>&1
$PY scale_run.py summary > logs/rev/scale_summary.log 2>&1

# 4. figures, tables, macros
$PY figures.py > logs/rev/figures10.log 2>&1
$PY fig_explain.py > logs/rev/fig_explain10.log 2>&1
mkdir -p generated logs/rev
$PY make_tables.py > logs/rev/tables10.log 2>&1
$PY make_rev_tables.py > logs/rev/revtables10.log 2>&1
echo FINAL_PIPELINE_DONE
