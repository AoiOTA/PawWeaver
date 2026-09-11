"""Render the bounded result; verify original hashes after all four episodes."""
import hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent
r=json.loads((O/'comparison.json').read_text());v=json.loads((O/'transformation_validation.json').read_text())
hashes=v['newW_oldN']['provenance']['source_hashes'];assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
(O/'original_hashes_after.json').write_text(json.dumps({'all_original_files_unchanged':True,'files':hashes},indent=2)+'\n')
lines=['# Normalizer exchange: four CPU closed-loop episodes','',
'New weights use amplitude25 actual STOP checkpoint 325 (326 completed iterations, 32,047,104 fresh transitions); old weights use neutral checkpoint 499. These are diagnostic derived deployment bundles, not trained checkpoints. Only normalizer `_mean`, `_var`, `_std`, `count` were exchanged. New/old source counts: '+str(v['newW_oldN']['provenance']['source_counts'])+'.','',
'Old normalization does not restore new-weight movement: forward tracking degrades and backward falls at 27.88 s. Old weights with new normalization retain most forward/backward response. This deployment intervention does not identify a unique training root cause. The actual final325 baseline differs from the earlier checkpoint250: its forward speed recovers but undesired turning remains.','',
'Final325 backward completes60 s with nonfoot contact throughout the post20 window; it is not clean supported standing. Old weights with new normalization have zero nonfoot contact in both post20 windows.', '', '## Command and end-effector evidence','',
'Commands are forward +0.20 / backward −0.15 m/s, zero lateral/yaw. Velocity means use complete target-time 8.00–59.98 s / resulting state 8.02–60.00 s. EE RMSE uses complete target-time 20.00–59.98 s / state20.02–60.00 s. Fall rows show fragments only; no complete-window metric is inferred.','',
'| W / N | Case | Actual seconds / fall | Mean vx, vy, yawdot | EE position / orientation RMSE |','|---|---|---|---|---|']
for name,run in r['runs'].items():
 for c in run['readout']['episodes']:
  w=c['windows']['steady_command'];h=c['windows']['pose_hold'];response=w.get('full_window_response',w.get('executed_fragment_response'));full=w['complete_no_fall_window'];m=h['full_window_metrics'] if full else h['executed_fragment_metrics']
  vals=', '.join(f'{x:.6f}' for x in response['actual_mean_vx_vy_yawdot'])
  ee=f"{m['ee_position_m']['rmse']:.6f} m / {m['ee_orientation_rad']['rmse']:.6f} rad" if m else 'none'
  suffix='' if full else ' (fragment)'
  lines.append(f"| {name} | {c['case_id']} | {c['actual_duration_s']:.2f} / {c['fallen']} | {vals}{suffix} | {ee}{suffix} |")
lines+=['','## Actual support and FK','',
'Existing E3 readout uses poststate20.00–60.00 s (2001 samples for complete runs), contact force >1 N per foot, nonfoot force >5 N, and actual nonfoot-ground contact pairs. FK applies existing MuJoCo hinge alignment offset −0.002 s. Fall row covers only post20.00–27.88 s. Full windows, per-foot contact fractions, foot heights and movement are retained in each support JSON.','',
'| W / N | Case | Foot-support count fractions 0 / 1 / 2 / 3 / 4 | Nonfoot force / ground-pair fraction | FK xyz max m / orientation max rad |','|---|---|---|---|---|']
for name,run in r['runs'].items():
 for c in run['support']['cases']:
  w=c['windows']['post20'];n=w['samples'];count=w['foot_net_force_gt1N_count_samples'];f=' / '.join(f'{count.get(str(i),0)/n:.4f}' for i in range(5));ck=c['alignment_checks']
  lines.append(f"| {name} | {c['case_id']} | {f} | {w['robot_nonfoot_net_force_gt5N_any_fraction']:.6f} / {w['nonfoot_ground_pairs']['any_fraction']:.6f} | {ck['fk_tcp_world_xyz_max_error_m']:.3g} / {ck['fk_tcp_world_orientation_max_error_rad']:.3g} |")
lines+=['','## Execution and validation','',
'`bash .../plan_v3_normalizer_closed_loop/run.sh prepare`, then `newW_oldN`, then `oldW_newN`; exact commands and raw logs have matching saved exitcodes, all 0. GPU hidden; CPU OMP/OpenBLAS/MKL single thread. Exactly four extra MuJoCo episodes, no PhysX, training, installs, commits or pushes. Baselines reuse amplitude25 initial/final MuJoCo dev11 reports and their saved full E3 readout/support; source hashes are retained.','',
'Preparation followed actual ExportedPolicy / bundle loader paths. A loaded derived bundle matches the corresponding checkpoint with identical in-memory normalizer exchange on 60 preserved old observations (30 each forward/backward), maximum action difference0 for both. All deployed tensors equal their assigned weights/statistics source; 279-input observation and original action mapping remain unchanged. Bundle manifest/policy hashes are recomputed and checked by load_bundle. Original bundle manifests, policies and checkpoints remain byte-identical after execution (original_hashes_after.json). Source NPZ case metadata and hashes are unchanged; only relative paths in the two-case subset manifest differ.','',
'`readout.py` reuses E3 summarize/support unchanged, binds support ROOT to this directory and loads the matching derived bundle. Readout exit0. `comparison.json` retains complete and fragment metrics separately, trace/report hashes and FK checks; no rerun was needed.','',
'B route only: one 18-joint Actor; EE follows current base XY and yaw with ground-fixed Z. Velocity commands come from the preset simulation trajectory; deployment operator source remains unimplemented. These two neutral tasks do not show low/high/lateral whole-body capability, world-fixed EE-only success, formal acceptance or hardware validity. Stop boundary reached; no automatic expansion.']
(O/'README.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
