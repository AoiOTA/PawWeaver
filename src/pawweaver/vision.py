"""RGB-D marker measurements. Ground-truth target poses are deliberately absent from this API."""
from dataclasses import dataclass
import heapq
import numpy as np
from .contracts import GoalSample

@dataclass(frozen=True)
class Intrinsics:
    width:int
    height:int
    fx:float
    fy:float
    cx:float
    cy:float
    min_depth:float
    max_depth:float

    def __post_init__(self):
        if not np.isfinite([self.fx,self.fy,self.cx,self.cy,self.min_depth,self.max_depth]).all():
            raise ValueError("Camera calibration must be finite")
        if min(self.width,self.height,self.fx,self.fy,self.min_depth)<=0 or self.max_depth<=self.min_depth:
            raise ValueError("Invalid calibrated camera geometry or depth bounds")

    @property
    def matrix(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1.]])

def quaternion_rotation(quaternion):
    """Explicit WXYZ calibration; normalize finite nonzero input."""
    q = np.asarray(quaternion,float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError("Expected finite nonzero WXYZ calibration quaternion")
    q = q/np.max(np.abs(q))
    w,x,y,z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def rotation_quaternion(rotation):
    import cv2
    vector = cv2.Rodrigues(np.asarray(rotation,float))[0].ravel()
    angle = np.linalg.norm(vector)
    return np.r_[np.cos(angle/2),vector*(.5 if angle < 1e-12 else np.sin(angle/2)/angle)]

def calibrated_transform(transform):
    transform = np.asarray(transform,float)
    if (transform.shape != (4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1])
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3,:3]),1,atol=1e-6)):
        raise ValueError("Expected calibrated rigid 4x4 transform")
    return transform

def transform_points(points,transform):
    transform = calibrated_transform(transform)
    return points@transform[:3,:3].T+transform[:3,3]

def register_depth(depth,depth_intr:Intrinsics,color_intr:Intrinsics,depth_to_color):
    """Explicit depth-to-color projection with a nearest-depth z-buffer (no direct pixel indexing)."""
    if depth.shape!=(depth_intr.height,depth_intr.width):
        raise ValueError("Depth dimensions differ from calibration")
    v,u = np.indices(depth.shape)
    valid = np.isfinite(depth)&(depth>=depth_intr.min_depth)&(depth<=depth_intr.max_depth)
    z = depth[valid]
    points = np.stack(((u[valid]-depth_intr.cx)*z/depth_intr.fx,
                       (v[valid]-depth_intr.cy)*z/depth_intr.fy,z),axis=-1)
    points = transform_points(points,depth_to_color)
    points = points[points[:,2]>0]
    uu = np.rint(color_intr.fx*points[:,0]/points[:,2]+color_intr.cx).astype(int)
    vv = np.rint(color_intr.fy*points[:,1]/points[:,2]+color_intr.cy).astype(int)
    inside = (uu>=0)&(uu<color_intr.width)&(vv>=0)&(vv<color_intr.height)
    registered = np.full(color_intr.height*color_intr.width,np.inf)
    np.minimum.at(registered,vv[inside]*color_intr.width+uu[inside],points[inside,2])
    registered[~np.isfinite(registered)]=np.nan
    return registered.reshape(color_intr.height,color_intr.width)

class MarkerEstimator:
    def __init__(self,color:Intrinsics,depth:Intrinsics,depth_to_color,marker_id:int,
                 marker_size_m:float,marker_to_goal,marker_to_goal_quat_wxyz):
        import cv2
        self.cv2 = cv2
        self.color,self.depth = color,depth
        self.depth_to_color = calibrated_transform(depth_to_color)
        self.marker_id,self.marker_size = marker_id,marker_size_m
        self.marker_to_goal = np.asarray(marker_to_goal,float)
        self.marker_to_goal_rotation = quaternion_rotation(marker_to_goal_quat_wxyz)
        if not np.isfinite(marker_size_m) or marker_size_m<=0 or self.marker_to_goal.shape!=(3,) or not np.isfinite(self.marker_to_goal).all():
            raise ValueError("Specify physical marker size and goal offset in marker frame")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())

    def measure(self,rgb,depth_m,capture_timestamp,world_from_color_at_capture):
        cv2 = self.cv2
        if rgb.shape!=(self.color.height,self.color.width,3):
            raise ValueError("RGB image dimensions differ from calibration")
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        corners,ids,_ = self.detector.detectMarkers(gray)
        if ids is None or self.marker_id not in ids:
            return None
        pixel_corners = corners[list(ids.flatten()).index(self.marker_id)].reshape(4,2)
        half = self.marker_size/2
        object_corners = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]],float)
        ok,rvec,tvec = cv2.solvePnP(object_corners,pixel_corners.astype(float),self.color.matrix,
                                  np.zeros(5),flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2,0]<=0:
            return None
        registered = register_depth(depth_m,self.depth,self.color,self.depth_to_color)
        u,v = np.rint(pixel_corners.mean(0)).astype(int)
        patch = registered[max(0,v-3):v+4,max(0,u-3):u+4]
        values = patch[np.isfinite(patch)]
        if len(values)<5:
            return None
        z = float(np.median(values))
        if not self.color.min_depth<=z<=self.color.max_depth or abs(z-tvec[2,0])>max(.03,.1*z):
            return None
        center = np.array([(u-self.color.cx)*z/self.color.fx,(v-self.color.cy)*z/self.color.fy,z])
        rotation,_ = cv2.Rodrigues(rvec)
        goal_c = center+rotation@self.marker_to_goal
        world_from_color_at_capture = calibrated_transform(world_from_color_at_capture)
        goal_w = transform_points(goal_c,world_from_color_at_capture)
        orientation = rotation_quaternion(world_from_color_at_capture[:3,:3]@rotation@self.marker_to_goal_rotation)
        confidence = float(np.exp(-np.std(values)/.01))
        return GoalSample(capture_timestamp,tuple(goal_w),tuple(orientation),True,confidence)

class DelayedMeasurements:
    def __init__(self,max_age=.5):
        self.queue=[]
        self.latest=None
        self.max_age=max_age
        self.sequence=0

    def enqueue(self,sample:GoalSample,delivery_time:float):
        if delivery_time<sample.timestamp:
            raise ValueError("Measurement cannot arrive before capture")
        self.sequence+=1
        heapq.heappush(self.queue,(delivery_time,self.sequence,sample))

    def update(self,now):
        while self.queue and self.queue[0][0]<=now:
            sample=heapq.heappop(self.queue)[2]
            if sample.valid and (self.latest is None or sample.timestamp>self.latest.timestamp):
                self.latest=sample
        return self.latest

    def hold_required(self,now):
        return self.latest is None or now-self.latest.timestamp>self.max_age
