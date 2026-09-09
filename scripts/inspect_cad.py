"""Audit an official STEP attachment without inventing a material density or mass."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID,TopAbs_SHELL

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("step",type=Path)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
raw=args.step.read_bytes()
text=raw.decode("utf-8",errors="replace")
reader=STEPControl_Reader()
if reader.ReadFile(str(args.step))!=IFSelect_RetDone:
    raise ValueError("OpenCascade could not read STEP attachment")
reader.TransferRoots()
shape=reader.OneShape()
box=Bnd_Box();BRepBndLib.Add_s(shape,box)
minimum,maximum=box.CornerMin(),box.CornerMax()
limits=(*minimum.Coord(),*maximum.Coord())
def count(kind):
    iterator=TopExp_Explorer(shape,kind);n=0
    while iterator.More():
        n+=1;iterator.Next()
    return n
properties=GProp_GProps();BRepGProp.VolumeProperties_s(shape,properties)
result={"source_page":"https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/glzd7a853owrmsk0?singleDoc",
    "attachment_path":"/attachments/yuque/0/2026/tar/29291030/1769335599868-b7b30f4f-a08a-4714-967c-e383b09f5d32.tar",
    "filename":args.step.name,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),
    "export_timestamp":re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",text).group(0),
    "products":re.findall(r"PRODUCT\s*\(\s*'([^']*)'",text),
    "reader_length_unit":"mm (OpenCascade default STEP target unit)",
    "bounds_mm":list(limits),"extent_mm":[limits[i+3]-limits[i] for i in range(3)],
    "solid_count":count(TopAbs_SOLID),"shell_count":count(TopAbs_SHELL),
    "signed_volume_mm3":properties.Mass(),
    "mass_records":len(re.findall(r"MASS_MEASURE|DENSITY|MASS_UNIT",text)),
    "resolved_mass_kg":None,"resolved_inertia_kg_m2":None,
    "conclusion":"CAD geometry is available. Volume is not mass; no density assumptions or mass redistribution applied."}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(result,indent=2,ensure_ascii=False))
