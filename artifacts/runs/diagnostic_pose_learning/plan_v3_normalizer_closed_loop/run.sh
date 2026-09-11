#!/usr/bin/env bash
set -euo pipefail
cd /home/lyb/pawweaver
O=artifacts/runs/diagnostic_pose_learning/plan_v3_normalizer_closed_loop
case "${1:?}" in prepare) cmd=(/home/lyb/miniconda3/envs/pawweaver-train/bin/python "$O/prepare.py");; newW_oldN|oldW_newN) cmd=(/home/lyb/miniconda3/envs/pawweaver-runtime/bin/python scripts/evaluate_commanded_mujoco.py --asset assets/generated/diagnostic --bundle "$O/${1}_bundle" --suite "$O/dev2" --output "$O/$1" --seed 0 --diagnostic --provisional-spec artifacts/runs/diagnostic_pose_learning/wbc_low_noise_learning/spec.json);; *) exit 2;; esac
cmd=(env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "${cmd[@]}")
[[ ! -e "$O/${1}_console.log" ]]
printf '%q ' "${cmd[@]}" > "$O/${1}_command.txt"
set +e
"${cmd[@]}" > "$O/${1}_console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$O/${1}_exitcode.txt"
exit "$code"
