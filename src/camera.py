#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import math

class Detection():

    def __init__(self, box_points, center, world_center, angle, color, hue, size):
        self.box_points = box_points
        self.center = center
        self.world_center = world_center
        self.angle = angle
        self.color = color
        self.hue = hue
        self.size = size

class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.array([[1,0,0,0], [0, -1, 0,350], [0, 0, -1, 1000], [0, 0, 0, 1]]) #Naive
        self.last_click = None

        self.last_click_world = None

        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        #self.grid_x_points = np.arange(-450, 500, 50)
        #self.grid_y_points = np.arange(-175, 525, 50)
        #self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        # AprilTags Info
        self.apriltag_points = dict()

        # Distortion Coefficients
        self.dist_coeffs = np.array([])

        # Projective Transport
        self.dest_points = [
            (390, 535), (365, 510), (415, 510), (415, 560), (365, 560),
            (890, 535), (865, 510), (915, 510), (915, 560), (865, 560),
            (890, 235), (865, 210), (915, 210), (915, 260), (865, 260),
            (390, 235), (365, 210), (415, 210), (415, 260), (365, 260),
            #(215, 610), (190, 585), (240, 585), (240, 635), (190, 635), #5
            #(1065, 110), (1040, 85), (1090, 85), (1090, 135), (1040, 135) #8
        ]

        # Grid Corner Points
        self.grid_points = []
        for x in range(-500, 501, 50):
            for y in range(-175, 476, 50):
                self.grid_points.append((x,y,0))

        self.src_points = None
        self.H_warp = np.eye(3)

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        image = self.VideoFrame.copy()
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        SQUARENESS = 1.6

        SMALL_AREA = 500
        LARGE_AREA = 3500
        
        h_approx = [125, 108, 95, 43, 17, 170]
        h_margins = [7,5,6,20,5,20]
        s_mins = [10, 10, 10, 80, 10, 10]
        s_maxs = [255, 255, 255, 255, 255, 255]
        v_mins = [25, 25, 150, 50, 25, 25, 55]
        v_maxs = [255, 255, 255, 255, 255, 255]
        colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple"]
        block_detections = []
        for i in range(len(h_approx)):
            #if i != 5: For testing specific color detection
            #    continue
            h = h_approx[i]
            color = colors[i]
            h_margin = h_margins[i]
            s_min = s_mins[i]
            s_max = s_maxs[i]
            v_min = v_mins[i]
            v_max = v_maxs[i]
            if h+h_margin >= 180:
                h_lower_max = h+h_margin-180
                lower_bound = np.array([h-h_margin, s_min, v_min])
                upper_bound = np.array([h+h_margin, s_max, v_max])
                lower_bound2 = np.array([0, s_min, v_min])
                upper_bound2 = np.array([h_lower_max, s_max, v_max])
                thresholded = cv2.inRange(hsv_image, lower_bound, upper_bound)
                thresholded2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
                thresholded = thresholded+thresholded2
            else:
                lower_bound = np.array([h-h_margin, s_min, v_min])
                upper_bound = np.array([h+h_margin, s_max, v_max])
                thresholded = cv2.inRange(hsv_image, lower_bound, upper_bound)

            large_kernel = np.ones((7,7), np.uint8)
            small_kernel = np.ones((5,5), np.uint8)
            filtered = cv2.erode(thresholded, small_kernel, iterations=1)
            filtered = cv2.dilate(filtered, small_kernel, iterations=1)
            filtered = cv2.erode(filtered, large_kernel, iterations=1)
            filtered = cv2.dilate(filtered, large_kernel, iterations=1)
            
            all_contours = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


            SIZE_THRESHOLD = 1500

            # For distractor objects in Task 3 Level 3
            # close_spaces = [(293, 456), (277, 276), (291, 383), (571, 201), (640, 347), (824, 289), (874,370)]
            # close_dist = 50

            for contours in all_contours:
                rect = rect_center, rect_size, rect_angle = cv2.minAreaRect(contours)
                rx, ry = rect_center
                if (rx < 200 or rx > 1100 or ry < 100 or ry > 600):
                    continue

                # # For distractor objects in Task 3 Level 3
                # close_enough = False
                # for cx, cy in close_spaces:
                #     dist = math.sqrt((rx - cx)**2 + (ry - cy)**2)
                #     if dist < close_dist:
                #         close_enough = True
                #         break
                # if not close_enough:
                #     continue


                rect_area = rect_size[0] * rect_size[1]
                if rect_area > SMALL_AREA and rect_area < LARGE_AREA and (max(rect_size) / min(rect_size)) < SQUARENESS:
                    # Pixel to World
                    warped_x, warped_y, warped_z = np.linalg.inv(self.H_warp) @ [rect_center[0], rect_center[1], 1] 
                    warped_x = int(warped_x / warped_z)
                    warped_y = int(warped_y / warped_z)
                    z = self.DepthFrameRaw[warped_y][warped_x]

                    inv_k = np.linalg.inv(self.intrinsic_matrix)
                    inv_h = np.linalg.inv(self.extrinsic_matrix) 
                    pixel_pos = np.array([[warped_x], [warped_y], [1]])
                    dot = np.dot(inv_k, pixel_pos)
                    camera_pos = z * dot
                    camera_pos = np.vstack((camera_pos, [1]))
                    world_pos = new_x, new_y, new_z, _ = inv_h @ camera_pos

                    if (rect_area < SIZE_THRESHOLD):
                        size = "Small"
                    else:
                        size = "Large"

                    rect_points = cv2.boxPoints(rect)
                    box = np.int0(rect_points)

                    detection = Detection(box, rect_center, world_pos, rect_angle, color, 0, size)

                    block_detections.append(detection)
        self.block_detections = np.array(block_detections)
        


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass


    # TODO: Grid is still off (but is aligned with world coordinates). Figure out where a warp needs to be corrected for
    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        
        modified_image = self.VideoFrame.copy()
        
        for grid_point in self.grid_points:
            gx, gy, gz = grid_point
            world_point = (gx, gy, gz, 1)
            camera_point = cx, cy, cz, _ = self.extrinsic_matrix @ world_point
            pixel_point = self.intrinsic_matrix @ [cx, cy, cz]
            pixel_point = [pixel_point[0] / pixel_point[2], pixel_point[1] / pixel_point[2], 1]
            pixel_point = self.H_warp @ pixel_point
            pixel_point = [pixel_point[0] / pixel_point[2], pixel_point[1] / pixel_point[2], 1]
            int_pixel_point = int(pixel_point[0]), int(pixel_point[1])
            cv2.circle(modified_image, int_pixel_point, 5, (0, 255, 0), -1)

        self.GridFrame = modified_image

        
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.blockDetector()
        for detection in self.block_detections:
            modified_pos = "X = " + str(int(detection.world_center[0])) + " Y = " + str(int(detection.world_center[1])) + " Z = " + str(int(detection.world_center[2]))
            cv2.drawContours(modified_image,[detection.box_points],-1,(0,0,255),2)
            cv2.putText(modified_image, "Color: " + str(detection.color) ,(int(detection.center[0]) - 5, int(detection.center[1]) - 5), font, 0.5, (0,0,255), 2)
            cv2.putText(modified_image, "Position: " + modified_pos,(int(detection.center[0]) - 5, int(detection.center[1]) - 25), font, 0.5, (0,0,255), 2)
            cv2.putText(modified_image, "Orientation: " + str(int(detection.angle)),(int(detection.center[0]) - 5, int(detection.center[1]) - 45), font, 0.5, (0,0,255), 2)
            cv2.putText(modified_image, "Size: " + str(detection.size),(int(detection.center[0]) - 5, int(detection.center[1]) - 65), font, 0.5, (0,0,255), 2)

        self.apriltag_points = dict()
        for detection in msg.detections:

            id = detection.id

            center_cood = detection.centre.x, detection.centre.y
            self.apriltag_points[id] = list()
            self.apriltag_points[id].append((detection.centre.x, detection.centre.y))

            proj_center_cood = self.projective_transform(center_cood)

            cv2.circle(modified_image, proj_center_cood, 5, (0, 255, 0), -1)

            int_corners = []
            for corner in detection.corners:
                self.apriltag_points[id].append((corner.x, corner.y))
                int_corners.append(self.projective_transform((corner.x, corner.y)))
            corners_array = np.array(int_corners, dtype=np.int32)
            cv2.polylines(modified_image, [corners_array], isClosed=True, color=(0,0,255), thickness = 2)

            cv2.putText(modified_image, "ID:"+str(detection.id), (proj_center_cood[0]+30, proj_center_cood[1]-30), font, 1, (255,0,0), 2)

        self.TagImageFrame = modified_image

    def projective_transform(self, point):
        point = [point[0], point[1], 1]
        output= self.H_warp @ point
        return int(output[0]/output[2]), int(output[1]/output[2])

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        if self.camera.src_points is not None:
            self.camera.H_warp = cv2.findHomography(np.array(self.camera.src_points), np.array(self.camera.dest_points))[0]
            cv_image = cv2.warpPerspective(cv_image, self.camera.H_warp, (cv_image.shape[1], cv_image.shape[0]))
        self.camera.VideoFrame = cv_image 


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        self.camera.dist_coeffs = np.reshape(data.d, (5, ))

class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)

        self.camera.DepthFrameRaw = cv_depth
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()