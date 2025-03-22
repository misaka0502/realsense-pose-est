# Intel RealSense Depth Camera compatible Python package for live 6 DOF pose estimation.
---
## Installation
This repository is extremely dependent on [FoundationPose](https://github.com/NVlabs/FoundationPose) and [Segment Anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file). Please follow the instructions of these two repositories to configure the environment.
Then, follow [librealsense Python warpper](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation) to install the packages for the Intel RealSense camera.

## Usage
- Run `python realsense_pose_est.py`.
- Then it will appears a window to show the color image captured by camera.
- Click the target object with the left mouse button, and the Mask obtained through SAM will be displayed. If you are not satisfied with the mask, you can close the mask window and click Get again; if you are satisfied with the mask, you can close the mask window and press ESC to exit.
    - **Note: After obtaining the mask, try not to move the camera and the object to avoid the problem of misalignment between the mask and the color image.**
- If everything goes well, the target pose estimation window will be displayed after a short while.
