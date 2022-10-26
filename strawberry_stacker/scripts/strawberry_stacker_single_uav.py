#!/usr/bin/env python3
from threading import Thread

import cv2
import numpy as np
import rospy
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError
from gazebo_ros_link_attacher.srv import Gripper
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from sensor_msgs.msg import Image
from std_msgs.msg import String, UInt8

rospy.init_node('strawberry_stacker_original_config', anonymous=True)


class DroneStateController:
    def __init__(self, drone: str):
        self.drone = drone

    def set_arm(self, arg: bool):
        """Arm/Disarm the Drone.
        Args:
            arg: boolean value -> True:Arm, False:Disarm
        """
        rospy.wait_for_service(f'/{self.drone}/mavros/cmd/arming')
        try:
            arm_service = rospy.ServiceProxy(
                f'/{self.drone}/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm_service(arg)
        except rospy.ServiceException as e:
            print("Service arming call failed: %s" % e)

    def set_mode(self, mode: str):
        """Changes the mode of Drone.
        Args:
            mode: string value -> custom_modes: ex:- 'OFFBOARD', 'AUTO.LAND', 'AUTO.TAKEOFF', 'AUTO.MISSION' etc.
        """
        rospy.wait_for_service(f'/{self.drone}/mavros/set_mode')
        try:
            modeService = rospy.ServiceProxy(
                f'/{self.drone}/mavros/set_mode', mavros_msgs.srv.SetMode)
            modeService(custom_mode=mode)
        except rospy.ServiceException as e:
            print("Service set mode call failed: %s" % e)


class GripperStateController:
    def __init__(self, drone: str):
        self.drone = drone
        self.gripper_range = String()

    def activate_gripper(self, arg: bool) -> bool:
        """Activate/Deactivate the Gripper.
        Args:
            arg: boolean value -> True:Activate, False:Deactivate
        Returns:
            Status of Gripper
        """
        rospy.wait_for_service(f'/{self.drone}/activate_gripper')
        try:
            gripper_service = rospy.ServiceProxy(
                f'/{self.drone}/activate_gripper', Gripper)
            gripper_status = gripper_service(arg)
            return gripper_status
        except rospy.ServiceException as e:
            print("Service set mode call failed: %s" % e)

    def check_range(self, status):
        """CallBack function for gripper_check topic"""
        self.gripper_range = status


class StateMoniter:
    def __init__(self):
        self.state = State()
        self.pose = PoseStamped()

    def state_call_back(self, msg):
        """CallBack function for /mavros/state topic"""
        self.state = msg

    def pose_call_back(self, msg):
        """CallBack function for /mavros/local_position/pose topic"""
        self.pose = msg


class ImageProcessor:
    def __init__(self, drone):
        self.drone = drone
        self.image_sub = rospy.Subscriber(
            f'/{self.drone}/camera/image_raw', Image, self.image_call_back)
        self.img = np.empty([])
        self.bridge = CvBridge()
        self.rate = rospy.Rate(50)

    def image_call_back(self, data):
        """CallBack function for /camera/image_raw topic"""
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def detect_aruco_marker(self, img: np.ndarray) -> dict:
        """Detects Aruco Marker from the image.
        Args:
            img: np.ndarray value -> image from which the Aruco Marker has to be detected
        Returns:
            Dictionary with Aruco ID as key and corners as value
        """
        detected_aruco_markers = {}
        dictionary = cv2.aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters = cv2.aruco.DetectorParameters_create()
        try:
            (corners, ids, _) = cv2.aruco.detectMarkers(
                img, dictionary, parameters=parameters)
            detected_aruco_markers = dict(zip(ids[:, 0], corners))
        except Exception:
            pass
        return detected_aruco_markers

    def calculate_aruco_marker_position(self, detected_aruco_markers: dict) -> float:
        """Detects Aruco Marker from the image.
        Args:
            Detected_ArUco_markers: Dictionary value -> Dictionary with Aruco ID as key and corners as value
        Returns:
            x coordinate of the center of Aruco Marker
        """
        for corners in detected_aruco_markers.values():
            for corner in corners:
                top_left = list(map(float, corner[0]))
                bottom_right = list(map(float, corner[2]))
                cx = (top_left[0] + bottom_right[0]) / 2

        return cx


class MultiDrone:
    def __init__(self, drone, edrone_rows, edrone_global_offset=0):
        self.drone = drone
        self.state_monitor = StateMoniter()
        self.drone_controller = DroneStateController(self.drone)
        self.gripper_monitor = GripperStateController(self.drone)
        self.aruco_processor = ImageProcessor(self.drone)
        self.rate = rospy.Rate(100)
        self.local_pose_publisher = rospy.Publisher(
            f'/{self.drone}/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.local_velocity_publisher = rospy.Publisher(
            f'/{self.drone}/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        rospy.Subscriber(f'/{self.drone}/mavros/state',
                         State, self.state_monitor.state_call_back)
        rospy.Subscriber(f'/{self.drone}/mavros/local_position/pose',
                         PoseStamped, self.state_monitor.pose_call_back)
        rospy.Subscriber(f'/{self.drone}/gripper_check',
                         String, self.gripper_monitor.check_range)
        rospy.Subscriber('/spawn_info', UInt8, self.box_row_no_callback)
        self.pose = PoseStamped()
        self.velocity = TwistStamped()
        self.edrone_global_offset = edrone_global_offset
        self.edrone_rows = edrone_rows
        self.home_pose = [self.state_monitor.pose.pose.position.x,
                          self.state_monitor.pose.pose.position.y, 0.5]
        self.box_pose = 0
        self.blue_box = 2
        self.red_box = 1
        self.boxid = 0
        self.pick_loc = []
        self.drop_loc = []
        self.red_truck_pose = [[60.45, 63.9], [60.45, 65.15], [60.45, 66.4], [59.6, 63.9], [59.6, 65.15], [59.6, 66.4],
                               [58.75, 63.9], [58.75, 65.15], [58.75, 66.4]]
        self.blue_truck_pose = [[17.4, -8.4], [17.4, -7.17, ], [17.4, -5.94], [16.54, -8.4], [16.54, -7.17], [16.54, -5.94],
                                [15.7, -8.4], [15.7, -7.17, ], [15.7, -5.94]]
        self.box_detected = False
        self.box_count_placed = 0
        self.box_count_picked = 0
        self.row_no = 0
        self.drone_reached_row_end = False

    def box_row_no_callback(self, msg):
        """Callback function for /spawn_info topic"""
        self.row_no = int(msg.data)
        if len(self.pick_loc) == 0:
            if self.drone == 'edrone0' and self.row_no % 3 == 2:
                self.edrone_rows[0][1] = (
                    4*(self.row_no-1))+self.edrone_global_offset
                self.pick_loc = [[self.state_monitor.pose.pose.position.x,
                                  self.state_monitor.pose.pose.position.y, 6], self.edrone_rows[0]]
            elif self.drone == 'edrone1' and self.row_no % 3 == 1:
                self.edrone_rows[0][1] = (
                    4*(self.row_no-1))+self.edrone_global_offset
                self.pick_loc = [[self.state_monitor.pose.pose.position.x,
                                  self.state_monitor.pose.pose.position.y, 6], self.edrone_rows[0]]

    def calculate_pick_location(self):
        """Calculates the pick location for next strawberry box"""
        self.edrone_rows[self.box_count_picked][1] = (
            4*(self.edrone_rows[self.box_count_picked][1]-1))+self.edrone_global_offset
        if self.boxid == self.blue_box:
            self.pick_loc = [[self.blue_truck_pose[self.box_count_placed][0], self.blue_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 6], [self.edrone_rows[self.box_count_picked][0], self.edrone_rows[self.box_count_picked][1], 6],
                             self.edrone_rows[self.box_count_picked]]
        elif self.boxid == self.red_box:
            self.pick_loc = [[self.red_truck_pose[self.box_count_placed][0], self.red_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 6], [self.edrone_rows[self.box_count_picked][0], self.edrone_rows[self.box_count_picked][1], 6],
                             self.edrone_rows[self.box_count_picked]]

    def calculate_drop_location(self):
        """Calculates the drop location for current strawberry box"""
        if self.boxid == self.blue_box:
            self.drop_loc = [[self.state_monitor.pose.pose.position.x, self.state_monitor.pose.pose.position.y, 6],
                             [self.blue_truck_pose[self.box_count_placed][0], self.blue_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 6], [self.blue_truck_pose[self.box_count_placed][0], self.blue_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 1.8]]
        elif self.boxid == self.red_box:
            self.drop_loc = [[self.state_monitor.pose.pose.position.x, self.state_monitor.pose.pose.position.y, 6],
                             [self.red_truck_pose[self.box_count_placed][0], self.red_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 6], [self.red_truck_pose[self.box_count_placed][0], self.red_truck_pose[self.box_count_placed][1]+self.edrone_global_offset, 1.8]]

    def set_arm(self, arg: bool):
        """Checks the state of the drone and Arm/Disarm the Drone accordingly. 
        Args:
            arg: boolean value -> True:Arm, False:Disarm
        """
        if arg:
            while not self.state_monitor.state.armed:
                self.drone_controller.set_arm(True)
                self.rate.sleep()
            print(f"{self.drone}:Armed!!")
        else:
            while self.state_monitor.state.armed:
                self.drone_controller.set_arm(False)
                self.rate.sleep()
            print(f"{self.drone}:Disarmed!!")

    def set_mode(self, mode):
        """Checks the current mode of the drone and Changes the mode of Drone accordingly.
        Args:
            mode: string value -> custom_modes: ex:- 'OFFBOARD', 'AUTO.LAND', 'AUTO.TAKEOFF', 'AUTO.MISSION' etc.
        """
        while not self.state_monitor.state.mode == mode:
            self.drone_controller.set_mode(mode)
            self.rate.sleep()
        print(f"{self.drone}:{mode} mode activated")

    def offboard_mode(self):
        """Changes the mode of the drone to offboard mode"""
        self.set_arm(True)
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 0
        for i in range(100):
            # Sending Dummy setpoints before switching to offboard mode
            self.local_pose_publisher.publish(self.pose)
            self.rate.sleep()
        self.set_mode('OFFBOARD')

    def strawberry_box_detector(self) -> bool:
        """Detects the position of the Stawberry box
            Returns:
                True if the drone is at the postion of Stawberry Box else returns False
        """
        Detected_ArUco_markers = self.aruco_processor.detect_aruco_marker(
            self.aruco_processor.img)
        try:
            cx = self.aruco_processor.calculate_aruco_marker_position(
                Detected_ArUco_markers)
        except:
            cx = 0
        if cx >= 190 and cx <= 200:  # checks whether the aruco marker center is in the range of the center of the image frame
            self.box_pose = self.state_monitor.pose.pose.position.x
            self.boxid = int(list(Detected_ArUco_markers.keys())[0])
            self.box_detected = True
            return True
        else:
            return False

    def gripper_control(self, is_pick: bool):
        """Checks the state of the Gripper and Activate/Deactivate the Gripper accordingly. 
        Args:
            is_pick: boolean value -> True:Activate, False:Deactivate
        """
        if is_pick:
            time_out = 0
            while self.gripper_monitor.gripper_range.data == 'False':
                time_out = time_out+1
                # Repositioning algorithms bychance if the gripper is not in the range
                if time_out > 200:
                    self.adjust_position(
                        [self.box_pose, self.edrone_rows[self.box_count_picked][1], 2], True)
                    self.adjust_position(
                        [self.box_pose, self.edrone_rows[self.box_count_picked][1], 0.1], False)
                    self.set_mode('AUTO.LAND')
                    time_out = 0
                self.rate.sleep()
            print(f'{self.drone}:gripper in range')
            while self.gripper_monitor.activate_gripper(True).result == False:
                self.rate.sleep()
            print(f'{self.drone}:Box picked')
        else:
            while self.gripper_monitor.activate_gripper(False).result == True:
                self.rate.sleep()
            print(f'{self.drone}:Box droped')

    def pick_box(self):
        """Function to pick the box using velocity setpoints with a velocity of 1m/s in x dir only"""
        self.velocity.twist.linear.x = 1
        self.velocity.twist.linear.y = 0
        self.velocity.twist.linear.z = 0
        while True:
            # publish velocity sentpoints untill the drone detects the strawberry box or reaches the row end
            self.local_velocity_publisher.publish(self.velocity)
            if round(self.state_monitor.pose.pose.position.x, 4) >= 59 and round(self.state_monitor.pose.pose.position.x, 4) <= 61:
                self.drone_reached_row_end = True
                self.velocity.twist.linear.x = 0
                self.velocity.twist.linear.y = 0
                self.velocity.twist.linear.z = 0
                self.local_velocity_publisher.publish(self.velocity)
                print("stoping")
                break
            # if box detected, then adjust the position of the drone onto the box and pick it
            elif self.strawberry_box_detector() == True:
                self.adjust_position(
                    [self.state_monitor.pose.pose.position.x, self.edrone_rows[self.box_count_picked][1], 0.1], False)
                self.set_mode('AUTO.LAND')
                self.gripper_control(True)
                return
            self.rate.sleep()

    def send_position_setpoints(self, setpoints: list):
        """Navigates the drone to the given position setpoint. 
        Args:
            setpoints: list of lists value -> ex:- [[0, 3.8, 3], [7, 0, 3], [16, 0, 3]]
        """
        for setpoint in setpoints:
            setpoint_reached = False
            self.pose.pose.position.x = setpoint[0]
            self.pose.pose.position.y = setpoint[1]
            self.pose.pose.position.z = setpoint[2]
            while not setpoint_reached:
                # publishes the postion setpoints untill the current setpoint is reached
                self.local_pose_publisher.publish(self.pose)
                if round(self.state_monitor.pose.pose.position.x, 4) >= setpoint[0]-0.1 and round(self.state_monitor.pose.pose.position.x, 4) <= setpoint[0]+0.1 and round(self.state_monitor.pose.pose.position.y, 4) >= setpoint[1]-0.1 and round(self.state_monitor.pose.pose.position.y, 4) <= setpoint[1]+0.1 and round(self.state_monitor.pose.pose.position.z, 4) >= setpoint[2]-0.1 and round(self.state_monitor.pose.pose.position.z, 4) <= setpoint[2]+0.1:
                    setpoint_reached = True
                self.rate.sleep()

    def adjust_position(self, setpoint: list, box: bool):
        """Adjusts the position of the drone to a particular position setpoint. 
        Args:
            setpoint: list value -> ex:- [7, 0, 3]
            box: boolean value -> True: used during repositioning of the drone for picking box, False: only adjust the position
        """
        if box:
            self.offboard_mode()
        self.pose.pose.position.x = setpoint[0]
        self.pose.pose.position.y = setpoint[1]
        self.pose.pose.position.z = setpoint[2]
        while True:
            self.local_pose_publisher.publish(self.pose)
            if round(self.state_monitor.pose.pose.position.x, 4) >= setpoint[0]-0.1 and round(self.state_monitor.pose.pose.position.x, 4) <= setpoint[0]+0.1 and round(self.state_monitor.pose.pose.position.y, 4) >= setpoint[1]-0.05 and round(self.state_monitor.pose.pose.position.y, 4) <= setpoint[1]+0.05 and round(self.state_monitor.pose.pose.position.z, 1) >= setpoint[2]-1 and round(self.state_monitor.pose.pose.position.z, 1) <= setpoint[2]+1:
                break
            self.rate.sleep()

    def pick(self, pick_loc: list):
        """performs pick operation. 
        Args:
            pick_loc: list of lists value -> ex:- [[0, 3.8, 3], [7, 0, 3], [16, 0, 3]]
        """
        self.offboard_mode()
        self.send_position_setpoints(pick_loc)
        self.pick_box()
        self.calculate_drop_location()

    def place(self, drop_loc: list):
        """performs drop operation. 
        Args:
            drop_loc: list of lists value -> ex:- [[0, 3.8, 3], [7, 0, 3], [16, 0, 3]]
        """
        self.offboard_mode()
        self.send_position_setpoints(drop_loc)
        self.set_mode('AUTO.LAND')
        self.gripper_control(False)

    def go_to_home(self, home_pose: list):
        """Makes the drone to come back to initial position. 
        Args:
            home_pose: list of lists value -> ex:- [[0, 3.8, 3], [7, 0, 3], [16, 0, 3]], initial position of the drone
        """
        self.offboard_mode()
        self.send_position_setpoints(home_pose)
        self.set_mode('AUTO.LAND')
        self.set_arm(False, True)

    def start_mission(self):
        """Starts the mission of picking and placing the starwberry boxes"""
        while len(self.pick_loc) == 0:
            self.rate.sleep()
        while self.box_count_picked < 30:
            if self.drone_reached_row_end:
                self.pick([self.pick_loc[1]])
                self.drone_reached_row_end = False
            else:
                self.pick(self.pick_loc)
                self.box_count_picked = self.box_count_picked + 1
            self.calculate_pick_location()
            if self.box_detected:
                self.place(self.drop_loc)
                self.box_count_placed = self.box_count_placed + 1
                self.box_detected = False
        self.go_to_home([self.home_pose])
        print("MISSION COMPLETED")


'''
edrone*_rows -> list of list in which : 1st value is x dir offset for the row, 2nd value is row number, 3rd value is altitude for the drone
edrone*_global_offset -> The global coordinates are considered with respect to edrone0, hence a particular offset value must be given to each drone to make its coordinates global i.e., w.r.t edrone0
'''


def main():
    """Main function for edrone0"""
    edrone0_rows = [[1, 1, 2], [1, 15, 2], [10, 15, 2],
                    [20, 15, 2], [40, 15, 2], [20, 11, 2], [20, 9, 2], [20, 13, 2]]
    edrone0 = MultiDrone('edrone0', edrone0_rows)
    edrone0.start_mission()




if __name__ == '__main__':
    # starts two threads for edrone0 and edrone1 respectively
    try:
        main()
    except rospy.ROSInterruptException:
        pass
