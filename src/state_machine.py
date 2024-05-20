"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from kinematicsIK import IK_geometric, adjust, FK_dh
import math

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.recorded_waypoints = []
        self.recorded_gripper_states = []
        self.apriltag_model_points = {
            1 : [(-250, -25, 0), (-275, 0, 0), (-225, 0, 0), (-225, -50, 0), (-275, -50, 0)],
            2 : [(250, -25, 0), (225, 0, 0), (275, 0, 0), (275, -50, 0), (225, -50, 0)],
            3 : [(250, 275, 0), (225, 300, 0), (275, 300, 0), (275, 250, 0), (225, 250, 0)],
            4 : [(-250, 275, 0), (-275, 300, 0), (-225, 300, 0), (-225, 250, 0), (-275, 250, 0)]
        }
        self.dh_params = np.array([[0, 0, 0, 0],
                      [0, -np.arctan(0.25), 103.91, 0],
                      [0, -np.arctan(4), 205.73, 0],
                      [0, 0, 200, 0],
                      [0, 0, 154.15, 0]
                      ])
        
        self.count = 0
        self.temp = [
            [2.04, 3.12, -2.12],
            [1.62, 3.08, -1.57]
        ]
        self.task_number = 4

        # Task Specific variables
        self.small_blocks = None
        self.large_blocks = None

        self.started_task2 = False
        self.current_depth = 0
        #self.candidate_positions = [(-205,23), (205, 23), (-205, 150), (205, 150), (-150, 230), (150, 230), (105,130)]
        self.candidate_positions = [(250,-25), (-250, -25)] # Stack 1 is LLS, Stack 2 is SLS
        self.first = True
        self.current_stack_position = None
        self.used_candidate_position = None
        self.blocks_stacked = 0


        # For task 4
        self.target_small = 0
        self.depth_small = 0
        self.target_large = 0
        self.depth_large = 0
        self.bad_detection = 0
        
    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record":
            self.record_waypoint()

        if self.next_state == "playback":
            self.playback()

        if self.next_state == "record_grip_true":
            self.record_grip_true()

        if self.next_state == "record_grip_false":
            self.record_grip_false()

        if self.next_state == "clear_waypoints":
            self.clear_waypoints()

        if self.next_state == "click_and_grab":
            self.click_and_grab()

        if self.next_state == "competition":
            self.competition()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"
        self.camera.blockDetector()

        # Check to see if the most recent click refers to a reasonable position to grab
        if self.camera.last_click_world is not None:
            x,y,z = self.camera.last_click_world
            x = x[0]
            y = y[0]
            z_grab = z[0] + 15
            z_release = z_grab + 30
            z_approach = z_grab + 100

            phi = adjust(np.arctan2(y,x) - np.pi)
            if np.linalg.norm([x,y]) > 370:
                y -= 20
                if x > 0:
                    x -= 20
                else:
                    x += 20
                theta = np.pi / 2
                psi = np.pi / 2
            else:
                theta = np.pi
                psi = -phi

            
            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)

            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)

            if any(math.isnan(x) for x in (config_approach + config_grab + config_release)):
                print("Location is not reachable")
                self.next_state = "idle"
                self.camera.last_click = None
                self.camera.last_click_world = None
            else:
                self.next_state = "click_and_grab"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        self.next_state = "idle"
        for waypoint in self.waypoints:    
            self.rxarm.set_positions(waypoint)
            time.sleep(4)

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        model_points = []
        image_points = []
        self.camera.src_points = []
        if len(self.apriltag_model_points.keys()) == len(self.camera.apriltag_points):
            for id in self.apriltag_model_points.keys():
                id_model_points = self.apriltag_model_points[id]
                id_img_points = self.camera.apriltag_points[id]
                model_points.extend(id_model_points)
                image_points.extend(id_img_points)
            for i in range(4):
                for j in range(5):
                    self.camera.src_points.append(self.camera.apriltag_points[i + 1][j])
            model_points = np.array(model_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            dist_coeffs = np.zeros((5,1))
            _, rot, trans = cv2.solvePnP(model_points, image_points, self.camera.intrinsic_matrix, dist_coeffs)
            rotational_matrix, _ = cv2.Rodrigues(rot)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotational_matrix
            extrinsic[:3, 3] = np.array(trans).reshape(3)
            self.camera.extrinsic_matrix = extrinsic
            self.status_message = "Calibration - Completed Calibration"
        else:
            self.status_message = "Calibration - Calibration Failed: Missing AprilTags"
            self.camera.src_points = None
            self.camera.apriltag_points = dict()


    def record_waypoint(self):
        self.status_message = "Recording Waypoint"
        self.current_state = "record"
        self.next_state = "idle"
        self.recorded_waypoints.append(self.rxarm.get_positions())

    def playback(self):

        self.status_message = "Playback"
        self.current_state = "playback"
        self.next_state = "idle"

        x_list = []
        y_list = []
        z_list = []
        m = FK_dh(self.dh_params, self.rxarm.get_positions(), 0)
        x, y, z = m[0:3, 3]
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        for i in range(len(self.recorded_waypoints)):
            waypoint = self.recorded_waypoints[i]
            gripper_status = self.recorded_gripper_states[i]
            self.rxarm.set_positions(waypoint)
            time.sleep(3)
            m = FK_dh(self.dh_params, self.rxarm.get_positions(), 0)
            x, y, z = m[0:3, 3]
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            if gripper_status:
                self.rxarm.gripper.grasp()
            else:
                self.rxarm.gripper.release()
            time.sleep(1)
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        z_list = np.array(z_list)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x_list, y_list, z_list)
        ax.scatter(x_list, y_list, z_list, c = plt.cm.jet(np.linspace(0,1,len(x_list))))
        plt.show()

    def record_grip_true(self):
        self.status_message = "Recording Gripper True"
        self.current_state = "record_grip_true"
        self.next_state = "idle"
        self.recorded_gripper_states.append(True)
        
        
    def record_grip_false(self):
        self.status_message = "Recording Gripper False"
        self.current_state = "record_grip_false"
        self.next_state = "idle"
        self.recorded_gripper_states.append(False)

    def clear_waypoints(self):
        self.status_message = "Clearing Waypoints"
        self.current_state = "clear_waypoints"
        self.next_state = "idle"
        self.recorded_waypoints = []
        self.recorded_gripper_states = []

    def click_and_grab(self):
        self.status_message = "Click and Grab"
        self.current_state = "click_and_grab"
        self.next_state = "idle"
        x,y,z = self.camera.last_click_world
        x = x[0]
        y = y[0]
        z_grab = z[0] + 25
        z_release = z_grab + 30
        z_approach = z_grab + 100

        phi = adjust(np.arctan2(y,x) - np.pi)
        theta = np.pi
        psi = -phi + np.pi/2
        pose_approach = [x,y,z_approach, phi, theta, psi]
        config_approach = IK_geometric(self.dh_params,pose_approach)
        if np.isnan(config_approach).any():
            # position is outside normal range, approach from side
            y -= 20
            if x > 0:
                x -= 20
            else:
                x += 20
            theta = np.pi *5/6
            psi = np.pi / 2

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)
            config_release = config_release * math.pi / 180
            config_release[0] = adjust(config_release[0] - np.pi)
            config_release[4] = adjust(config_release[4] - np.pi)

        else:
            # position is inside normal range, approach from above
            theta = np.pi
            psi = -phi + np.pi/2

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)
            config_release = config_release * math.pi / 180
            config_release[0] = adjust(config_release[0] - np.pi)
            config_release[4] = adjust(config_release[4] - np.pi)
    
        self.rxarm.set_positions(config_approach)
        time.sleep(2)
        if self.count % 2 == 0:
            self.rxarm.set_positions(config_grab)
            time.sleep(2)
            self.rxarm.gripper.grasp()
        else:
            self.rxarm.set_positions(config_release)
            time.sleep(2)
            self.rxarm.gripper.release()
        time.sleep(1)
        self.rxarm.set_positions(config_approach)
        time.sleep(2)
        self.camera.last_click = None
        self.camera.last_click_world = None
        self.count += 1

    
    def grab_position(self, position, isGrab):
        x = position[0]
        y = position[1]
        z_grab = position[2] + 25
        z_release = z_grab + 30
        z_approach = z_grab + 100

        phi = adjust(np.arctan2(y,x) - np.pi)
        theta = np.pi
        psi = -phi + np.pi/2
        pose_approach = [x,y,z_approach, phi, theta, psi]
        config_approach = IK_geometric(self.dh_params,pose_approach)
        if np.isnan(config_approach).any():
            # position is outside normal range, approach from side
            y -= 20
            if x > 0:
                x -= 20
            else:
                x += 20
            theta = np.pi *5/6
            psi = np.pi / 2

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)
            config_release = config_release * math.pi / 180
            config_release[0] = adjust(config_release[0] - np.pi)
            config_release[4] = adjust(config_release[4] - np.pi)

        else:
            # position is inside normal range, approach from above
            theta = np.pi
            psi = -phi + np.pi/2

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)
            config_release = config_release * math.pi / 180
            config_release[0] = adjust(config_release[0] - np.pi)
            config_release[4] = adjust(config_release[4] - np.pi)
        
        self.rxarm.set_positions(config_approach)
        time.sleep(2)
        if isGrab:
            self.rxarm.set_positions(config_grab)
            time.sleep(2)
            self.rxarm.gripper.grasp()
        else:
            self.rxarm.set_positions(config_release)
            time.sleep(2)
            self.rxarm.gripper.release()
        time.sleep(1)
        self.rxarm.set_positions(config_approach)
        time.sleep(2)

    def grab_detection(self, detection, isGrab):
        print("Attempting to grab block at ", detection.world_center)
        x = detection.world_center[0]
        y = detection.world_center[1]
        if detection.size == "Large":
            z_grab = detection.world_center[2] + 15
        else:
            z_grab = detection.world_center[2] + 20
        z_release = z_grab + 30
        z_approach = z_grab + 100

        phi = adjust(np.arctan2(y,x) - np.pi)

        canGrab = self.can_grab_from_above(detection)
        if not canGrab:
            y -= 20
            if x > 0:
                x -= 20
            else:
                x += 20
            theta = np.pi * 5/6
            psi = np.pi / 2

            distance_up = 200

            d = math.sqrt(x**2 + y**2)
            a = math.asin(y / d)
            up_x = math.cos(a) * (d - distance_up)
            up_y = math.sin(a) * (d - distance_up)

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_up = [up_x,up_y,z_approach, phi, theta, psi]
            config_up = IK_geometric(self.dh_params,pose_up)
            config_up = config_up * math.pi / 180
            config_up[0] = adjust(config_up[0] - np.pi)
            config_up[4] = adjust(config_up[4] - np.pi)

        else:
            # position is inside normal range, approach from above
            theta = np.pi
            psi = -phi - (detection.angle * np.pi / 180)

            pose_approach = [x,y,z_approach, phi, theta, psi]
            config_approach = IK_geometric(self.dh_params,pose_approach)
            config_approach = config_approach * math.pi / 180
            config_approach[0] = adjust(config_approach[0] - np.pi)
            config_approach[4] = adjust(config_approach[4] - np.pi)
            
            pose_grab = [x,y,z_grab, phi, theta, psi]
            config_grab = IK_geometric(self.dh_params,pose_grab)
            config_grab = config_grab * math.pi / 180
            config_grab[0] = adjust(config_grab[0] - np.pi)
            config_grab[4] = adjust(config_grab[4] - np.pi)

            pose_release = [x,y,z_release, phi, theta, psi]
            config_release = IK_geometric(self.dh_params,pose_release)
            config_release = config_release * math.pi / 180
            config_release[0] = adjust(config_release[0] - np.pi)
            config_release[4] = adjust(config_release[4] - np.pi)
        
        if canGrab:
            self.rxarm.set_positions(config_approach)
            time.sleep(2)
            if isGrab:
                self.rxarm.set_positions(config_grab)
                time.sleep(2)
                self.rxarm.gripper.grasp()
            else:
                self.rxarm.set_positions(config_release)
                time.sleep(2)
                self.rxarm.gripper.release()
            time.sleep(1)
            self.rxarm.set_positions(config_approach)
            time.sleep(2)
        else:
            self.rxarm.set_positions(config_approach)
            time.sleep(2)
            self.rxarm.set_positions(config_grab)
            time.sleep(2)
            self.rxarm.gripper.grasp()
            time.sleep(1)
            self.rxarm.set_positions(config_approach)
            time.sleep(2)

    def distance(self, detection1, detection2=None):
        if detection2 != None:
            return math.sqrt((detection1.world_center[0] - detection2.world_center[0])**2 + (detection1.world_center[1] - detection2.world_center[1])**2)
        else:
            return np.linalg.norm([detection1.world_center[0], detection1.world_center[1]])

    def can_grab_from_above(self, detection):
        x = detection.world_center[0]
        y = detection.world_center[1]
        if detection.size == "Large":
            z_grab = detection.world_center[2] + 15
        else:
            z_grab = detection.world_center[2] + 20
        z_approach = z_grab + 100

        phi = adjust(np.arctan2(y,x) - np.pi)
        theta = np.pi
        psi = -phi
        pose_approach = [x,y,z_approach, phi, theta, psi]
        config_approach = IK_geometric(self.dh_params,pose_approach)
        return not np.isnan(config_approach).any()

    def task1(self):
        self.status_message = "Competition Task 1"
        y_cutoff = 0
        x_offset = 150
        if self.small_blocks is None and self.large_blocks is None:
            print("Starting Task 1")
            self.large_blocks = 0
            self.small_blocks = 0
        if any(detection.world_center[1] > y_cutoff for detection in self.camera.block_detections):
            print(len(list(detection for detection in self.camera.block_detections if detection.world_center[1] > y_cutoff)), " detections valid")
            valid_detections = [detection for detection in self.camera.block_detections if detection.world_center[1] > y_cutoff]
            min_dist = 99999
            current_detection = None
            for detection in valid_detections:
                dist = self.distance(detection)
                if dist < min_dist:
                    min_dist = dist
                    current_detection = detection

            # Edge case: If a small block is on top of a large one, make sure to grab the small one first
            if current_detection.size == "Large":
                for detection in list(detection for detection in self.camera.block_detections if detection.world_center[1] > y_cutoff):
                    if detection.size == "Small":
                        dist = self.distance(detection, current_detection)
                        # dist = math.sqrt((detection.world_center[0] - current_detection.world_center[0])**2 + (detection.world_center[1] - current_detection.world_center[1])**2)
                        if dist < 15:
                            # Change current_detection to the small one
                            current_detection = detection
                            break

            self.grab_detection(current_detection, True)
            #print("Grabbing Block at ", current_detection.world_center)
            if current_detection.size == "Small":
                new_position = [-x_offset - (self.small_blocks % 3)*50, -50 - (50*int(self.small_blocks/3)), -20]
                self.small_blocks += 1
            else:
                new_position = [x_offset + (self.large_blocks % 3)*50, -50 - (50*int(self.large_blocks/3)), -10]
                self.large_blocks += 1
            print("Placing ", current_detection.size ," block at ", new_position)
            self.grab_position(new_position, False)
            self.next_state = "competition"
            print(self.small_blocks, " small blocks sorted, ", self.large_blocks, " large blocks sorted.")
        else:
            print("Task 1 Complete")
            self.small_blocks = None
            self.large_blocks = None
            self.next_state = "idle"
    
    def task3(self):
        self.status_message = "Competition Task 3"

        y_cutoff = 0
        x_offset = 125

        color_sequence = ["Red",
                          "Orange",
                          "Yellow",
                          "Green",
                          "Blue",
                          "Purple"]

        if any(detection.world_center[1] > y_cutoff for detection in self.camera.block_detections):
            valid_detections = [detection for detection in self.camera.block_detections if detection.world_center[1] > y_cutoff]
            min_dist = 99999
            current_detection = None
            for detection in valid_detections:
                dist = self.distance(detection)
                if dist < min_dist:
                    min_dist = dist
                    current_detection = detection

            # Edge case: If a small block is on top of a large one, make sure to grab the small one first
            if current_detection.size == "Large":
                for detection in list(detection for detection in self.camera.block_detections if detection.world_center[1] > y_cutoff):
                    if detection.size == "Small":
                        dist = self.distance(detection, current_detection)
                        # dist = math.sqrt((detection.world_center[0] - current_detection.world_center[0])**2 + (detection.world_center[1] - current_detection.world_center[1])**2)
                        if dist < 15:
                            # Change current_detection to the small one
                            current_detection = detection
                            break

            self.grab_detection(current_detection, True)

            index = color_sequence.index(current_detection.color)
            if current_detection.size == "Small":
                new_position = [-x_offset - index*29, -50, -20]
            else:
                new_position = [x_offset + index*44, -50, -10]

            self.grab_position(new_position, False)
            self.next_state = "competition"
            
        else:
            print("Task 3 Complete")
            self.next_state = "idle"
        
    def task2(self):
        self.status_message = "Competition Task 2"


        if self.current_stack_position is None:
            
            for candidate in self.candidate_positions:
                cx, cy = candidate
                is_valid = True
                for detection in self.camera.block_detections:
                    dx, dy, dz, _ = detection.world_center
                    dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
                    if dist < 70:
                        is_valid = False
                        break
                if is_valid:
                    self.current_stack_position = candidate

        cx, cy = self.current_stack_position
        is_valid = False
        for detection in (detection for detection in self.camera.block_detections if detection.size == "Large"):
            is_below = False
            for detection2 in (detection for detection in self.camera.block_detections):
                if self.distance(detection, detection2) < 35 and detection2.world_center[2] - detection.world_center[2] > 10:
                    is_below = True
                    break
            if is_below:
                continue
            dx, dy, dz, _ = detection.world_center
            dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
            if self.used_candidate_position is not None:
                used_dist = math.sqrt((self.used_candidate_position[0] - dx)**2 + (self.used_candidate_position[1] - dy)**2)
            else:
                used_dist = 9999
            if dist >= 70 and used_dist >= 70:
                is_valid = True
                break
                # At this point, we have a valid large block to stack
        if not is_valid:
            for detection in (detection for detection in self.camera.block_detections if detection.size == "Small"):
                is_below = False
                for detection2 in (detection for detection in self.camera.block_detections):
                    if self.distance(detection, detection2) < 35 and detection2.world_center[2] - detection.world_center[2] > 10:
                        is_below = True
                        break
                if is_below:
                    continue
                dx, dy, dz, _ = detection.world_center
                dist = math.sqrt((cx - dx)**2 + (cy - dy)**2)
                if self.used_candidate_position is not None:
                    used_dist = math.sqrt((self.used_candidate_position[0] - dx)**2 + (self.used_candidate_position[1] - dy)**2)
                else:
                    break
        # Place detection onto candidate position
        self.grab_detection(detection, True)
        if detection.size == "Small":
            self.grab_position([cx, cy, self.current_depth - 13], False)
        else:
            self.grab_position([cx, cy, self.current_depth - 13], False)
              
        self.blocks_stacked += 1
        if detection.size == "Small":
            self.current_depth += 25
        else:
            self.current_depth += 40
        if self.blocks_stacked % 3 == 0:
            self.used_candidate_position = self.current_stack_position
            self.current_stack_position = None
            self.current_depth = 0
        if self.blocks_stacked >= 6:
            print("Task 2 Complete")
            self.used_candidate_positions = None
            self.next_state = "idle"
        else:
            self.next_state = "competition"
        
    
    def task4(self):


        self.status_message = "Competition Task 4"

        print(self.target_small, self.target_large)

        color_sequence = ["Red",
                          "Orange",
                          "Yellow",
                          "Green",
                          "Blue",
                          "Purple"]
        
        # target_small, depth_small = 0, 0
        # target_large, depth_large = 0, 0
        # bad_detection = 0

        lx, ly = 250, 25
        sx, sy = -250, 25

        min_dist = 99999
        current_detection, closest_detection = None, None
        for detection in self.camera.block_detections:
            dist = self.distance(detection)
            l_dist = math.sqrt((lx-detection.world_center[0]) ** 2 + (ly-detection.world_center[1]) ** 2)
            s_dist = math.sqrt((sx-detection.world_center[0]) ** 2 + (sy-detection.world_center[1]) ** 2)
            if dist < min_dist and l_dist > 30 and s_dist > 30:
                min_dist = dist
                closest_detection = detection   

        # cx, cy = self.current_stack_position
        # is_valid = False
        for detection in (detection for detection in self.camera.block_detections if detection.size == "Large"):
            if color_sequence.index(detection.color) == self.target_large:
                current_detection = detection
                break

        for detection in (detection for detection in self.camera.block_detections if detection.size == "Small"):
            if color_sequence.index(detection.color) == self.target_small:
                current_detection = detection
                break

        if current_detection is not None:
            # Place detection onto candidate position
            if self.can_grab_from_above(current_detection):

                self.grab_detection(detection, True)

                if detection.size == "Small":
                    print("Placing", current_detection.size, self.target_small)
                    self.grab_position([sx, sy, self.depth_small - 22], False)
                    self.target_small += 1
                    self.depth_small += 28
                else:
                    print("Placing", current_detection.size, self.target_small)
                    self.grab_position([lx, ly, self.depth_large - 13], False)
                    self.target_large += 1
                    self.depth_large += 40
            else:
                self.grab_detection(detection, True)
                # self.grab_detection(closest_detection, True)
            
                new_position = [100 + self.bad_detection*45, 75 - (50*int(self.bad_detection/3)), -10]
                self.bad_detection += 1

                self.grab_position(new_position, False)

        else:
            
            self.grab_detection(closest_detection, True)
            
            new_position = [100 + self.bad_detection*45, 75 - (50*int(self.bad_detection/3)), -10]
            self.bad_detection += 1

            self.grab_position(new_position, False)

        ## Possibly add depth detection at stack position to confirm success stack  


        if self.target_large == 6 and self.target_small == 6:
            print("Task 4 Complete")
            self.next_state = "idle"
        else:
            # self.target_small, self.depth_small = 0, 0
            # self.target_large, self.depth_large = 0, 0
            # self.bad_detection = 0
            self.next_state = "competition"


    def competition(self):
        
        self.current_state = "competition"
        
        if self.task_number == 1:
            self.task1()
        elif self.task_number == 2:
            self.task2()
        elif self.task_number == 3:
            self.task3()
        elif self.task_number == 4:
            self.task4()
        elif self.task_number == 5: #BONUS
            pass


    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)