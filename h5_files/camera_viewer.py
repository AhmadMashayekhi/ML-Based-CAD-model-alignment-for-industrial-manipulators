## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

## Edited by Menthy Denayer 2022-2023

###############################################
##      Open CV and Numpy integration        ##
###############################################

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import math

"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""
def gyro_to_angle(gyro,g_angles):
    sample_rate = 200 #Hz
    delta_t = 1/sample_rate
    
    g_angles = g_angles + gyro*delta_t
    
    return g_angles

def accel_to_angle(accel):
    acc_x= accel[0]
    acc_y = accel[1]
    acc_z = accel[2]
    
    sqrt_x = math.sqrt(acc_y*acc_y + acc_z*acc_z)
    angle_x = math.atan(acc_x,sqrt_x)
    
    sqrt_y = math.sqrt(acc_x*acc_x + acc_z*acc_z)
    angle_y = math.atan(acc_y,sqrt_y)
    
    sqrt_z = math.sqrt(acc_y*acc_y + acc_x*acc_x)
    angle_z = math.atan(sqrt_z,acc_z)
    
    a_angles = np.array([angle_x,angle_y,angle_z])
    
    return a_angles

def gyro_data(gyro, g_angles):
    gyro = np.asarray([gyro.x, gyro.y, gyro.z])
    print("gyro: ", gyro)
    
    angles = gyro_to_angle(gyro, g_angles)
    
    print("gyro angles: ", angles)
    
    return gyro, angles


def accel_data(accel):
    accel = np.asarray([accel.x, accel.y, accel.z])
    print("accelerometer: ", accel)
    
    angles = accel_to_angle(accel)
    
    print("accelerometer angles: ", angles)
    
    return accel

"""
=============================================================================
--------------------------------CONFIGURATION--------------------------------
=============================================================================
"""

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable gyroscope/accelerometer streams
# config.enable_stream(rs.stream.accel)
# config.enable_stream(rs.stream.gyro)

g_angles = np.array([0,0,0])

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(":: Device product line is: ", device_product_line)

# Setting RGB Info
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Set resolution
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

if device_product_line == 'D400':
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

#Background substraction (replace Images with fgMask)
# backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv2.createBackgroundSubtractorKNN()

# Setting High Accuracy preset #########################################################
# depth_sensor = pipe_profile.get_device().first_depth_sensor()

# preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
# #print('preset range:'+str(preset_range))_
# for i in range(int(preset_range.max)):
#     visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
#     print('%02d: %s' %(i,visulpreset))
#     if visulpreset == "High Accuracy":
#         depth_sensor.set_option(rs.option.visual_preset, i)
########################################################################################

"""
=============================================================================
----------------------------------START LOOP---------------------------------
=============================================================================
"""

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Read accelerometer & gyroscope data
        # accel = accel_data(frames[2].as_motion_frame().get_motion_data())
        # gyro, angles = gyro_data(frames[3].as_motion_frame().get_motion_data(),g_angles)
        
        #g_angles =  angles
        
        # Read depth data
        depth_frame = frames.get_depth_frame()
        
        # Read color data
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        
        #remove background
        #fgMask = backSub.apply(images)
        
        # Show images & draw rectangle
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (540,260)
  
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (740, 460)
          
        # Blue color in BGR
        color = (255, 0, 0)
          
        # Line thickness of 2 px
        thickness = 2
          
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(images, start_point, end_point, color, thickness)
        #image = cv2.circle(images, (300,200),200,(0,255,0),2)
        
        cv2.imshow('RealSense',image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()