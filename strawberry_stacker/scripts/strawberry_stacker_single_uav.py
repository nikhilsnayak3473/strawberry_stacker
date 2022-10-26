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

rospy.init_node('strawberry_stacker_single_uav', anonymous=True)


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
    def __init__(self, drone):
        self.drone = drone
        self.state_monitor = StateMoniter()
        self.drone_controller = DroneStateController(self.drone)
        self.gripper_monitor = GripperStateController(self.drone)
        self.aruco_processor = ImageProcessor(self.drone)
        self.rate = rospy.Rate(100)
        self.local_pose_publisher = rospy.Publisher(
            f'/{self.drone}/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.local_velocity_publisher = rospy.Publisher(
            f'/{self.drone}/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        rospy.Subscriber(f'/{self.drone}/mavros/state',
                         State, self.state_monitor.state_call_back)
        rospy.Subscriber(f'/{self.drone}/mavros/local_position/pose',
                         PoseStamped, self.state_monitor.pose_call_back)
        rospy.Subscriber(f'/{self.drone}/gripper_check',
                         String, self.gripper_monitor.check_range)
        rospy.Subscriber('/spawn_info', UInt8, self.box_row_no_callback)
        self.pose = PoseStamped()
        self.velocity = TwistStamped()
        self.strawberry_box_rows = {}
        self.home_pose = [self.state_monitor.pose.pose.position.x,
                          self.state_monitor.pose.pose.position.y, 0.5]
        self.box_pose_x = 0
        self.blue_box = 2
        self.red_box = 1
        self.boxid = 0
        self.box_pose = []
        self.pick_loc = []
        self.drop_loc = []
        self.red_truck_pose = [[60.45, 63.9], [60.45, 65.15], [60.45, 66.4], [59.6, 63.9], [59.6, 65.15], [59.6, 66.4],
                               [58.75, 63.9], [58.75, 65.15], [58.75, 66.4]]
        self.blue_truck_pose = [[17.4, -8.4], [17.4, -7.17, ], [17.4, -5.94], [16.54, -8.4], [16.54, -7.17], [16.54, -5.94],
                                [15.7, -8.4], [15.7, -7.17, ], [15.7, -5.94]]
        self.box_detected = False
        self.red_box_count_placed = 0
        self.blue_box_count_placed = 0
        self.box_count_picked = 0
        self.drone_reached_row_end = False

    def box_row_no_callback(self, msg):
        row_no = msg.data
        if row_no in self.strawberry_box_rows.keys():
            current_box_count = self.strawberry_box_rows.get(row_no)
            self.strawberry_box_rows[row_no] = current_box_count+1
        else:
            self.strawberry_box_rows[row_no] = 0

        self.box_pose.append(
            [10*self.strawberry_box_rows[row_no], 4*(int(row_no)-1), 2])

    def calculate_pick_location(self):
        if self.boxid == self.blue_box:
            self.pick_loc = [[self.blue_truck_pose[self.blue_box_count_placed][0], self.blue_truck_pose[self.blue_box_count_placed][1], 6], [self.box_pose[self.box_count_picked][0], self.box_pose[self.box_count_picked][1], 6],
                             self.box_pose[self.box_count_picked]]
        elif self.boxid == self.red_box:
            self.pick_loc = [[self.red_truck_pose[self.red_box_count_placed][0], self.red_truck_pose[self.red_box_count_placed][1], 6], [self.box_pose[self.box_count_picked][0], self.box_pose[self.box_count_picked][1], 6],
                             self.box_pose[self.box_count_picked]]

    def calculate_drop_location(self):
        if self.boxid == self.blue_box:
            self.drop_loc = [[self.state_monitor.pose.pose.position.x, self.state_monitor.pose.pose.position.y, 6],
                             [self.blue_truck_pose[self.blue_box_count_placed][0], self.blue_truck_pose[self.blue_box_count_placed][1], 6], [self.blue_truck_pose[self.blue_box_count_placed][0], self.blue_truck_pose[self.blue_box_count_placed][1], 1.8]]
        elif self.boxid == self.red_box:
            self.drop_loc = [[self.state_monitor.pose.pose.position.x, self.state_monitor.pose.pose.position.y, 6],
                             [self.red_truck_pose[self.red_box_count_placed][0], self.red_truck_pose[self.red_box_count_placed][1], 6], [self.red_truck_pose[self.red_box_count_placed][0], self.red_truck_pose[self.red_box_count_placed][1], 1.8]]

    def set_arm(self, arg: bool):
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
        while not self.state_monitor.state.mode == mode:
            self.drone_controller.set_mode(mode)
            self.rate.sleep()
        print(f"{self.drone}:{mode} mode activated")

    def offboard_mode(self):
        self.set_arm(True)
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 0
        for i in range(100):
            self.local_pose_publisher.publish(self.pose)
            self.rate.sleep()
        self.set_mode('OFFBOARD')

    def strawberry_box_detector(self) -> bool:
        Detected_ArUco_markers = self.aruco_processor.detect_aruco_marker(
            self.aruco_processor.img)
        try:
            cx = self.aruco_processor.calculate_aruco_marker_position(
                Detected_ArUco_markers)
        except:
            cx = 0
        if cx >= 190 and cx <= 200:
            self.box_pose_x = self.state_monitor.pose.pose.position.x
            self.boxid = int(list(Detected_ArUco_markers.keys())[0])
            self.box_detected = True
            return True
        else:
            return False

    def gripper_control(self, is_pick: bool):
        if is_pick:
            time_out = 0
            while self.gripper_monitor.gripper_range.data == 'False':
                time_out = time_out+1
                if time_out > 200:
                    self.adjust_position(
                        [self.box_pose_x, self.box_pose[self.box_count_picked][1], 2], True)
                    self.adjust_position(
                        [self.box_pose_x, self.box_pose[self.box_count_picked][1], 0.1], False)
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
        self.velocity.twist.linear.x = 1
        self.velocity.twist.linear.y = 0
        self.velocity.twist.linear.z = 0
        while True:
            self.local_velocity_publisher.publish(self.velocity)
            if round(self.state_monitor.pose.pose.position.x, 4) >= 59 and round(self.state_monitor.pose.pose.position.x, 4) <= 61:
                self.drone_reached_row_end = True
                self.velocity.twist.linear.x = 0
                self.velocity.twist.linear.y = 0
                self.velocity.twist.linear.z = 0
                self.local_velocity_publisher.publish(self.velocity)
                print("stoping")
                break
            elif self.strawberry_box_detector() == True:
                self.adjust_position(
                    [self.state_monitor.pose.pose.position.x, self.box_pose[self.box_count_picked][1], 0.1], False)
                self.set_mode('AUTO.LAND')
                self.gripper_control(True)
                return
            self.rate.sleep()

    def send_position_setpoints(self, setpoints: list):
        for setpoint in setpoints:
            setpoint_reached = False
            self.pose.pose.position.x = setpoint[0]
            self.pose.pose.position.y = setpoint[1]
            self.pose.pose.position.z = setpoint[2]
            while not setpoint_reached:
                self.local_pose_publisher.publish(self.pose)
                if round(self.state_monitor.pose.pose.position.x, 4) >= setpoint[0]-0.1 and round(self.state_monitor.pose.pose.position.x, 4) <= setpoint[0]+0.1 and round(self.state_monitor.pose.pose.position.y, 4) >= setpoint[1]-0.1 and round(self.state_monitor.pose.pose.position.y, 4) <= setpoint[1]+0.1 and round(self.state_monitor.pose.pose.position.z, 4) >= setpoint[2]-0.1 and round(self.state_monitor.pose.pose.position.z, 4) <= setpoint[2]+0.1:
                    setpoint_reached = True
                self.rate.sleep()

    def adjust_position(self, setpoint: list, box: bool):
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
        self.offboard_mode()
        self.send_position_setpoints(pick_loc)
        self.pick_box()
        self.calculate_drop_location()

    def place(self, drop_loc: list):
        self.offboard_mode()
        self.send_position_setpoints(drop_loc)
        self.set_mode('AUTO.LAND')
        self.gripper_control(False)

    def go_to_home(self, home_pose: list):
        self.offboard_mode()
        self.send_position_setpoints(home_pose)
        self.set_mode('AUTO.LAND')
        self.set_arm(False, True)

    def start_mission(self):
        while len(self.box_pose) == 0:
            self.rate.sleep()

        self.pick_loc = [[self.state_monitor.pose.pose.position.x,
                          self.state_monitor.pose.pose.position.y, 6], self.box_pose[0]]
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
                if self.boxid == self.blue_box:
                    self.blue_box_count_placed = self.blue_box_count_placed + 1
                else:
                    self.red_box_count_placed = self.red_box_count_placed + 1
                self.box_detected = False
        self.go_to_home([self.home_pose])
        print("MISSION COMPLETED")


def main():
    edrone0 = MultiDrone('edrone0')
    edrone0.start_mission()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
