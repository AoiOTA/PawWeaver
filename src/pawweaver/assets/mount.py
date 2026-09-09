"""A visible mounting concept derived from the manual rail spacing and the source shell.

Geometry only: this is not a fabricated part, measured assembly, or inertial calibration.
"""
import numpy as np
import xml.etree.ElementTree as ET
from .model import format_vec

def surface_height(mesh,xy):
    triangles=np.asarray(mesh.triangles)
    a=triangles[:,1,:2]-triangles[:,0,:2]
    b=triangles[:,2,:2]-triangles[:,0,:2]
    determinant=a[:,0]*b[:,1]-b[:,0]*a[:,1]
    good=np.abs(determinant)>1e-12
    triangles,a,b,determinant=triangles[good],a[good],b[good],determinant[good]
    delta=np.asarray(xy)-triangles[:,0,:2]
    u=(delta[:,0]*b[:,1]-b[:,0]*delta[:,1])/determinant
    v=(a[:,0]*delta[:,1]-delta[:,0]*a[:,1])/determinant
    inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
    z=triangles[:,0,2]+u*(triangles[:,1,2]-triangles[:,0,2])+v*(triangles[:,2,2]-triangles[:,0,2])
    if not inside.any():
        raise ValueError("Mount support lies outside the source base shell")
    return float(z[inside].max())

def preview_mount(base_link,mesh):
    # Manual: rail spacing 152 mm, each rail 140 x 20 x 18 mm; attachment stations +/-60 and 0 mm.
    # The central shell surface, not the unrelated front sensor maximum, sets the mounting plane.
    shell=surface_height(mesh,[0,0])
    rail_bottom=shell
    rail_top=rail_bottom+.018
    plate_thickness=.008  # Proposed adapter dimension, NOT an official supplied component.
    def box(name,xyz,size,color):
        visual=ET.SubElement(base_link,"visual",name=name)
        ET.SubElement(visual,"origin",xyz=format_vec(xyz),rpy="0 0 0")
        ET.SubElement(ET.SubElement(visual,"geometry"),"box",size=format_vec(size))
        ET.SubElement(ET.SubElement(visual,"material",name=name),"color",rgba=color)
    supports=[]
    for side,y in (("left",.076),("right",-.076)):
        box("preview_rail_"+side,[0,y,rail_bottom+.009],[.14,.02,.018],"0.35 0.38 0.43 1")
        for i,x in enumerate((-.06,0.,.06)):
            z=surface_height(mesh,[x,y])
            height=rail_bottom-z
            if height>0:
                box(f"preview_seat_{side}_{i}",[x,y,z+height/2],[.012,.012,height],"0.22 0.25 0.3 1")
            supports.append({"xyz_m":[x,y,z],"height_m":max(0,height)})
    box("preview_adapter_plate",[0,0,rail_top+plate_thickness/2],[.14,.18,plate_thickness],"0.12 0.45 0.65 1")
    mount=[0.,0.,rail_top+plate_thickness]
    return mount,{"shell_center_z_m":shell,"rail_spacing_m":.152,"rail_size_m":[.14,.02,.018],
        "proposed_plate_size_m":[.14,.18,plate_thickness],"supports":supports,
        "mass_and_collision":"Visual-only concept; excluded from inertia and training until assembly is verified",
        "arm_base_plane_z_m":mount[2]}
