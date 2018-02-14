## OpenCV Python Stereo Camera Calibration

This repository contains some sources to calibrate the intrinsics of individual cameras and also the extrinsics of a stereo pair using OpenCV. It is an attempt to calibrate a generic camera pair as well as RGB-D cameras. Stores calibration data in YAML format. 


### Dependencies

- OpenCV
- NumPy
- YAML


### Example of RGBD camera calibration
python3 calibrate.py


### Stereo calibration for extrinisics

Once you have the intrinsics calibrated for both the left and the right cameras, you can use their intrinsics to calibrate the extrinsics between them.


For example, if you calibrated the left and the right cameras using the images in the `calib_imgs/1/` directory, the following command to compute the extrinsics.


### Undistortion and Rectification

Once you have the stereo calibration data, you can remove the distortion and rectify any pair of images so that the resultant epipolar lines become scan lines.


 
### See Also
- ROS based calibrator: http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
- Visual Inertial Calibrator vis-calib: https://github.com/arpg/vicalib
- The concept is trivial but some discussion for Kinect is in these two papers: 
	http://www.mas.bg.ac.rs/_media/istrazivanje/fme/vol43/1/8_bkaran.pdf
	http://www.cse.oulu.fi/~dherrera/papers/2012-KinectJournal.pdf

- Related: https://github.com/erget/StereoVision/tree/master/stereovision
