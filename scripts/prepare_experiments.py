"""Write reviewable three-seed curriculum/ablation jobs. This command starts no training."""
import argparse
import itertools
import json
from pathlib import Path
from pawweaver.trajectories import Trajectory

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=Path("artifacts/experiments"))
parser.add_argument("--asset",type=Path,default=Path("assets/generated/verified"))
parser.add_argument("--iterations",type=int,default=1000,help="Per-stage engineering budget; validation decides adequacy")
parser.add_argument("--demonstrations",type=Path,nargs="*",default=[])
args=parser.parse_args()
for path in args.demonstrations:
    if Trajectory.load(path).metadata.get("split")!="train":
        raise ValueError("Experiment demonstrations must already belong to train split")
base=json.loads(Path("configs/training.json").read_text())
args.output.mkdir(parents=True,exist_ok=True)
jobs=[]
for prediction,adaptive,demos in itertools.product((False,True),(False,True),(False,True) if args.demonstrations else (False,)):
    variant=f"prediction{int(prediction)}_adaptive{int(adaptive)}_demonstrations{int(demos)}"
    for seed in (0,1,2):
        previous=None
        for stage in range(4):
            directory=args.output/variant/f"seed{seed}"/f"stage{stage}"
            directory.mkdir(parents=True,exist_ok=True)
            config=dict(base,trajectory_prediction=prediction,adaptive_sampling=adaptive,curriculum_stage=stage,
                domain_randomization=stage==3,demonstrations=[str(p.resolve()) for p in args.demonstrations] if demos and stage==3 else [])
            path=directory/"config.json";path.write_text(json.dumps(config,indent=2)+"\n")
            argv=[str(Path.home()/"miniconda3/envs/pawweaver-train/bin/python"),"scripts/train.py","--asset",str(args.asset),
                "--config",str(path),"--output",str(directory/"run"),"--seed",str(seed),"--iterations",str(args.iterations),"--headless"]
            if previous:
                argv.extend(["--initialize-from",previous])
            jobs.append({"variant":variant,"seed":seed,"stage":stage,"argv":argv,
                         "requires_previous_stage_validation":stage>0})
            previous=str(directory/"run"/f"checkpoint_{args.iterations-1:06d}.pt")
manifest={"schema_version":1,"jobs":jobs,"demonstration_ablation_available":bool(args.demonstrations),
    "status":"prepared_not_executed","promotion":"Evaluate fixed suite after each stage; advance only after stable behavior. No automatic promotion from training reward."}
(args.output/"jobs.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Prepared {len(jobs)} jobs; demonstration ablation available: {bool(args.demonstrations)}")
