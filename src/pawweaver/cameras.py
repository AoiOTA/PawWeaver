"""Calibrated pinhole camera parameters, tested against rendered pixel coordinates."""
def mujoco_intrinsics(intrinsics):
    width,height=intrinsics.width,intrinsics.height
    # MuJoCo's principalpixel is image-plane shift, not OpenCV's absolute pixel coordinate.
    # Renderer samples pixel centers at half offsets; OpenCV uses integer pixel centers.
    return {"resolution":f"{width} {height}","sensorsize":"1 1",
        "focalpixel":f"{intrinsics.fx} {intrinsics.fy}",
        "principalpixel":f"{(width-1)/2-intrinsics.cx} {(height-1)/2-intrinsics.cy}"}
