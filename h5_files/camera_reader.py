## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

#Link: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

#2 foto's anders?
"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#misc imports
from H5_writer import uniquify

"""
=============================================================================
--------------------------------CONFIGURATION--------------------------------
=============================================================================
"""

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(":: Device product line is: ", device_product_line) 
#D455 stereo depth resoltion of 1280x720, range: 0.4m to 6m

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("No color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #640, 480

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
if device_product_line == 'D400':
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
    
# Start streaming
pipe_profile = pipeline.start(config)
rs.pointcloud()

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
--------------------------------START CAPTURE--------------------------------
=============================================================================
"""

try:
    for i in range(5):
        frames = pipeline.wait_for_frames()
        
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    # Filters #################################################################
    #dec_filter = rs.decimation_filter()    # Decimation - reduces depth frame density
    spat_filter = rs.spatial_filter()      # Spatial    - edge-preserving spatial smoothing
    #temp_filter = rs.temporal_filter()     # Temporal   - reduces temporal noise

    #depth_frame = dec_filter.process(depth_frame)
    depth_frame = spat_filter.process(depth_frame)
    #depth_frame = temp_filter.process(depth_frame)
    ###########################################################################
    
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    #print(depth_image.shape)
    color_image = np.asanyarray(color_frame.get_data())
    #print(color_image.shape)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))
        
    """-------------SHOW IMAGES-------------"""
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # Start coordinate, here (220,140)
    # represents top of the rectangle (should be in the middle)
    start_point = (540,260)

    # Ending coordinate, here (420, 340)
    # represents the bottom right corner of rectangle
    end_point = (740, 460)
      
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2
      
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(images, start_point, end_point, color, thickness)
    
    cv2.imshow('RealSense', images)
    cv2.waitKey(1) #Wait 1ms for key interuption

finally:
    # Stop streaming
    pipeline.stop()

"""
=============================================================================
-----------------------------PROCESSING RESULTS------------------------------
=============================================================================
"""

"""-------------PLOTS-------------"""
#Show color data
plt.figure(figsize=(14,3))
color = np.asanyarray(color_frame.get_data())
plt.rcParams["axes.grid"] = False
plt.subplot(141)
plt.imshow(color)

#Show depth camera
colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
plt.subplot(142)
plt.imshow(colorized_depth)

"""-------------Point Cloud Creation-------------"""
#Create Point Cloud
pc = rs.pointcloud();
pc.map_to(color_frame);
pointcloud = pc.calculate(depth_frame);

#Export point cloud (total) to .ply file
DIR = os.path.join(os.getcwd()+"/h5_files/camera/result.ply")
DIR = uniquify(DIR)
pointcloud.export_to_ply(DIR, color_frame);
print(":: Saved as: ", DIR)