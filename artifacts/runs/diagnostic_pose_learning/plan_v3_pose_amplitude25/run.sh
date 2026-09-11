#!/usr/bin/env bash
# Run one explicitly selected phase; root must hand off simulations first.
set -euo pipefail
cd /home/lyb/pawweaver
O=artifacts/runs/diagnostic_pose_learning/plan_v3_pose_amplitude25
E=artifacts/runs/diagnostic_pose_learning/plan_v3_commanded_pose
N=artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning
R=/home/lyb/miniconda3/envs/pawweaver-runtime/bin/python
T=/home/lyb/miniconda3/envs/pawweaver-train/bin/python
S=artifacts/runs/diagnostic_pose_learning/wbc_low_noise_learning/spec.json
phase=${1:?phase required}
case "$phase" in
train500)
 cmd=(env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 "$T" scripts/train.py --asset assets/generated/diagnostic --diagnostic --provisional-spec "$S" --config "$O/config.json" --initialize-from "$N/train500/checkpoint_000499.pt" --output "$O/train500" --seed 0 --num-envs 4096 --iterations 500 --stop-file "$O/STOP" --headless);;
export250)
 cmd=(env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$T" "$E/export_checkpoint.py" --checkpoint "$O/train500/checkpoint_000250.pt" --output "$O/checkpoint250_bundle");;
initial_mujoco_dev11|initial_physx_dev11|checkpoint250_mujoco_dev11|final_mujoco_dev11|final_physx_dev11)
 bundle="$O/train500/bundle"; suite="$O/dev11"; count=11
 [[ "$phase" != initial_* ]] || bundle="$N/train500/bundle"
 [[ "$phase" != checkpoint250_* ]] || bundle="$O/checkpoint250_bundle"
 cmd=(env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1)
 if [[ "$phase" == *_physx_* ]]; then cmd+=("$T" scripts/evaluate_commanded_isaac.py --num-envs "$count" --headless); else cmd+=(CUDA_VISIBLE_DEVICES= "$R" scripts/evaluate_commanded_mujoco.py); fi
 cmd+=(--asset assets/generated/diagnostic --bundle "$bundle" --suite "$suite" --output "$O/$phase" --seed 0 --diagnostic --provisional-spec "$S");;
*) echo "Unknown phase: $phase" >&2; exit 2;;
esac
[[ ! -e "$O/${phase}_console.log" ]] || { echo 'Refusing to overwrite first execution log' >&2; exit 2; }
printf '%q ' "${cmd[@]}" > "$O/${phase}_command.txt"
set +e
"${cmd[@]}" > "$O/${phase}_console.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$O/${phase}_exitcode.txt"
exit "$status"
