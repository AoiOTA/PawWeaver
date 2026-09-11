"""Use unchanged E3 support readout with this experiment's suite directory."""
from pathlib import Path
import argparse,importlib.util,json
O=Path(__file__).resolve().parent
source=O.parent/'plan_v3_commanded_pose/support_readout.py'
s=importlib.util.spec_from_file_location('e3_support',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.ROOT=O
p=argparse.ArgumentParser()
for name in ('evaluation','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();r=m.build(a.evaluation,a.bundle);r['suite_root_override']=str(O);r['invocation_source']=str(Path(__file__).resolve());a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'cases':len(r['cases']),'max_fk_error':max(v['alignment_checks']['fk_tcp_world_xyz_max_error_m'] for v in r['cases'])}))
