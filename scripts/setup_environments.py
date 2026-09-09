#!/usr/bin/env python3
"""Reproducible Conda-only setup. Never install packages into base or the source Isaac environment."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAB_REVISION = "ffff603eafc6b74264a5261cc0183d6a65390d78"


def run(args, **kwargs):
    clean = os.environ.copy()
    clean.pop("PYTHONPATH", None)  # Do not inject globally sourced ROS plugins.
    clean["PYTHONNOUSERSITE"] = "1"
    subprocess.run([str(arg) for arg in args], check=True, env=clean, cwd=ROOT, **kwargs)


def apply_metadata_compatibility(lab: Path):
    """Resolve two contradictory upstream pins in this exact tag; no simulator algorithm changes."""
    replacements = [("isaaclab", '"coverage==7.6.1",', '"coverage==7.4.4",'),
                    ("isaaclab_rl", '"packaging<24",', '"packaging==26.0",')]
    for package, old, new in replacements:
        path = lab/"source"/package/"setup.py"
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old,new))
        elif new not in text:
            raise RuntimeError(f"Upstream metadata changed: {path}")
        fragment = lab/"source"/package/"changelog.d/pawweaver-sim601-compat.skip"
        fragment.parent.mkdir(parents=True,exist_ok=True)
        fragment.write_text("Local PawWeaver dependency metadata compatibility with installed Isaac Sim 6.0.1.\n")


def setup(conda: Path, role: str, source_env: str):
    base = Path(json.loads(subprocess.check_output([str(conda),"info","--json"]))["root_prefix"])
    envs = json.loads(subprocess.check_output([str(conda),"env","list","--json"]))["envs"]
    name = "pawweaver-"+role
    prefix = base/"envs"/name
    if str(prefix) not in envs:
        create = [conda,"create","-y","-q","-n",name]
        run(create+["--clone",source_env] if role == "train" else create+["python=3.12","pip"])
    python = prefix/"bin/python"
    if prefix == base or prefix.name == source_env:
        raise ValueError("Refusing to modify base or source environment")
    if role == "data":
        run([python,"-m","pip","install","-e",".[data]"])
    elif role == "runtime":
        run([python,"-m","pip","install","torch==2.11.0","--index-url","https://download.pytorch.org/whl/cpu"])
        run([python,"-m","pip","install","-e",".[control,test,vision,render]"])
    else:
        lab = ROOT/".deps/IsaacLab"
        if not lab.exists():
            env = os.environ.copy()
            env["GIT_LFS_SKIP_SMUDGE"] = "1"
            subprocess.run(["git","clone","--depth","1","--branch","v3.0.0-beta2.patch1",
                            "https://github.com/isaac-sim/IsaacLab.git",str(lab)],check=True,env=env)
        revision = subprocess.check_output(["git","-C",str(lab),"rev-parse","HEAD"],text=True).strip()
        if revision != LAB_REVISION:
            raise ValueError("Unexpected Isaac Lab revision")
        apply_metadata_compatibility(lab)
        packages = ["isaaclab", "isaaclab_physx", "isaaclab_ov", "isaaclab_rl[rsl-rl]"]
        command = [python,"-m","pip","install","-e",".[control,test,data,vision,render,train]"]
        for package in packages:
            command.extend(["-e",str(lab/"source"/package)])
        command.extend(["--extra-index-url","https://pypi.nvidia.com"])
        run(command)
    run([python,"-m","pip","check"])
    record = ROOT/"artifacts/environments"
    record.mkdir(parents=True,exist_ok=True)
    with (record/f"{name}.freeze.txt").open("w") as stream:
        run([python,"-m","pip","freeze"],stdout=stream)
    with (record/f"{name}.conda.yml").open("w") as stream:
        run([conda,"env","export","-n",name,"--no-builds"],stdout=stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda",type=Path,default=Path.home()/"miniconda3/bin/conda")
    parser.add_argument("--role",choices=["train","runtime","data","all"],default="all")
    parser.add_argument("--isaac-env",default="isaacsim")
    args = parser.parse_args()
    for role in (["train","runtime","data"] if args.role == "all" else [args.role]):
        setup(args.conda,role,args.isaac_env)
