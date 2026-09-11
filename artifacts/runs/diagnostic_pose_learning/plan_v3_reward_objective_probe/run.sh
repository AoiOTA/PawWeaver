#!/usr/bin/env bash
set -euo pipefail
cd /home/lyb/pawweaver
O=artifacts/runs/diagnostic_pose_learning/plan_v3_reward_objective_probe
phase=${1:?old or new}
attempt=${2:-1}
log_name=${phase}_attempt${attempt}
case "$phase" in
old) B=artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/train500/bundle;;
new) B=artifacts/runs/diagnostic_pose_learning/plan_v3_pose_amplitude25/train500/bundle;;
*) exit 2;;
esac
cmd=(env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python "$O/evaluate_reward.py" --asset assets/generated/diagnostic --bundle "$B" --suite "$O/suite" --output "$O/$phase" --seed 0 --num-envs 3 --headless --diagnostic --provisional-spec artifacts/runs/diagnostic_pose_learning/wbc_low_noise_learning/spec.json)
[[ ! -e "$O/${log_name}_console.log" ]] || exit 2
printf '%q ' "${cmd[@]}" > "$O/${log_name}_command.txt"
set +e
"${cmd[@]}" > "$O/${log_name}_console.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$O/${log_name}_exitcode.txt"
exit "$status"
