#!/usr/bin/env python3
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import hypot

from gs_viewpoint_generator.msg import view_points_msg

class ViewpointGenerator:
    def __init__(self):
        self.resolution = 0.125
        self.map_width = 20
        self.map_height = 20
        self.map = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        self.exp = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        self.known_img = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)], np.uint8)

        self.safe_dis = 1.0

        self.pose = PoseStamped().pose

        self.map_sub = rospy.Subscriber('/uav1/exploration_node/sdf_map/occupancy_all', PointCloud2, self.map_callback)
        self.odom_sub = rospy.Subscriber('/uav1/mavros/local_position/pose', PoseStamped, self.odom_callback)

        self.traj_history = []
        self.close_dis = 0.2
        self.view_points_pub = rospy.Publisher('/uav1/exploration_node/view_points', view_points_msg, queue_size=10)
        self.view_points_msg = view_points_msg()

    def odom_callback(self, msg: PoseStamped):
        self.pose = msg
        for traj_point in self.traj_history:
            if hypot(traj_point.pose.position.x - self.pose.pose.position.x, traj_point.pose.position.y - self.pose.pose.position.y) < self.close_dis:
                return
        self.traj_history.append(self.pose)

    def map_callback(self, msg: PointCloud2):
        self.view_points_msg = view_points_msg()
        self.map = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        self.exp = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        self.known_img = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)], np.uint8)

        # 两张地图，一张是地形地图，一张是已知范围地图
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            self.exp[int(point[0] / self.resolution + self.map_width/self.resolution/2), int(point[1] / self.resolution + self.map_height/self.resolution/2)] = 255
            if point[2] < 0.5:
                continue
            # print(int((point[0] + self.map_width/2) / self.resolution), int((point[1] +self.map_height/2) / self.resolution))
            # self.map[int((point[0] + self.map_width/2) / self.resolution), int((point[1] +self.map_height/2) / self.resolution)] = 255
            # print(int(point[0] / self.resolution + self.map_width/self.resolution/2), int(point[1] / self.resolution + self.map_height/self.resolution/2))
            self.map[int(point[0] / self.resolution + self.map_width/self.resolution/2), int(point[1] / self.resolution + self.map_height/self.resolution/2)] = 255
            # print(point)

        # 可视化地图
        vis_img = self.map.copy()
        vis_img[vis_img==255] = 100
        vis_img[vis_img==0] = 255
        vis_img[vis_img==100] = 0

        # 选取轮廓
        mask = (self.map == 255) & (self.exp == 255)
        self.known_img[mask] = 255
        
        # 膨胀
        kernel = np.ones((int(self.safe_dis/self.resolution), int(self.safe_dis/self.resolution)), np.uint8)
        kernel_small = np.ones((int(0.4/self.resolution), int(0.4/self.resolution)), np.uint8)
        self.exp = cv2.dilate(self.exp, kernel_small, iterations=1).astype(np.uint8)
        dilation = cv2.dilate(self.known_img, kernel, iterations=1).astype(np.uint8)
        mask = (dilation == 255) & (self.exp == 255)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        line = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        line_tmp = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)])
        cv2.drawContours(line_tmp, contours, -1, 255, 1)
        mask = (line_tmp == 255) & (self.exp == 255)
        line[mask] = 255

        points = []
        def get_list(point1, point2):
            # global points
            x1, y1 = point1
            x2, y2 = point2
            dif_x = x2 - x1
            dif_y = y2 - y1
            dif = hypot(dif_x, dif_y)
            num = dif // (0.5 / self.resolution)
            for i in range(int(num)):
                x = x1 + dif_x * i / num
                y = y1 + dif_y * i / num
                points.append([int(x), int(y)])
                # print(points)
        # print(contours)
        for contour in contours:
            for num in range(len(contour)):
                # if contour[num][0][0] < 100 or contour[num][0][1] < 100:
                #     continue
                # if contour[num][0][0] > 900 or contour[num][0][1] > 900:
                #     continue
                if num + 1 < len(contour):
                    get_list(contour[num][0], contour[num+1][0])
                else:
                    get_list(contour[num][0], contour[0][0])
        # tsp_points = []
        view_points = []
        
        vis = np.ones([int(self.map_width/self.resolution), int(self.map_height/self.resolution), 3], np.uint8) * 100
        vis[self.map==255] = [0, 0, 0]
        mask = (self.map == 0) & (self.exp == 255)
        vis[mask] = [255, 255, 255]
        vis[line==255] = [255, 0, 0]
        for point in points:
            if self.exp[point[1], point[0]] == 255 and self.map[point[1], point[0]] == 0:
                cv2.circle(vis, (point[0], point[1]), 1, (0, 0, 255), -1)
                # print(point)
                # tsp_points.append(point)
                view_points.append(point)
        
        # 计算视场
        view_left = []
        view_right = []
        FOV = 70/180*np.pi
        view_length = 1.0*np.tan(FOV/2)/self.resolution
        colors = [[0, 0, 225], [0, 225, 0], [255, 0, 0]]
        garbage_points = []
        for i in range(len(view_points)):
            if len(view_points) < 2:
                continue
            if i == 0:
                yaw = np.arctan2(view_points[i+1][1]-view_points[i][1], view_points[i+1][0]-view_points[i][0])
            elif i == len(view_points)-1:
                yaw = np.arctan2(view_points[i][1]-view_points[i-1][1], view_points[i][0]-view_points[i-1][0])
            else:
                yaw = np.arctan2(view_points[i+1][1]-view_points[i-1][1], view_points[i+1][0]-view_points[i-1][0])
            yaw -= np.pi/2

            view_point_tmp1l = [int(view_points[i][0]+view_length*np.cos(yaw+FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw+FOV/2))]
            view_point_tmp1r = [int(view_points[i][0]+view_length*np.cos(yaw-FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw-FOV/2))]

            view_point_tmp2l = [int(view_points[i][0]+view_length*np.cos(yaw+FOV/2+np.pi)), int(view_points[i][1]+view_length*np.sin(yaw+FOV/2+np.pi))]
            view_point_tmp2r = [int(view_points[i][0]+view_length*np.cos(yaw-FOV/2+np.pi)), int(view_points[i][1]+view_length*np.sin(yaw-FOV/2+np.pi))]

            img_tmp1 = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)], np.uint8)
            img_tmp2 = np.zeros([int(self.map_width/self.resolution), int(self.map_height/self.resolution)], np.uint8)

            cv2.fillPoly(img_tmp1, [np.array([view_points[i], view_point_tmp1l, view_point_tmp1r])], 255)
            cv2.fillPoly(img_tmp2, [np.array([view_points[i], view_point_tmp2l, view_point_tmp2r])], 255)

            num1 = np.sum(img_tmp1 & self.known_img)
            num2 = np.sum(img_tmp2 & self.known_img)
            if num1 < num2:
                yaw += np.pi

            if num1 == 0 and num2 == 0:
                # view_points.remove(point)
                continue

            view_left = [int(view_points[i][0]+view_length*np.cos(yaw+FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw+FOV/2))]
            view_right = [int(view_points[i][0]+view_length*np.cos(yaw-FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw-FOV/2))]
            vis_tmp = vis.copy()
            cv2.fillPoly(vis_tmp, [np.array([view_points[i], view_left, view_right])], colors[i%3])
            vis = cv2.addWeighted(vis, 0.5, vis_tmp, 0.5, 0)

            valid = True
            for point in self.traj_history:
                if hypot(point.pose.position.x - view_points[i][0]*self.resolution, point.pose.position.y - view_points[i][1]*self.resolution) < 0.2:
                    valid = False
                    break
            if valid:
                point_tmp = PoseStamped()
                point_tmp.pose.position.x = view_points[i][0]*self.resolution
                point_tmp.pose.position.y = view_points[i][1]*self.resolution
                point_tmp.pose.orientation.z = yaw
                self.view_points_msg.view_points.append(point_tmp)
        
        self.view_points_pub.publish(self.view_points_msg)

        vis = cv2.resize(vis, (1012, 1012))
        vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('view line', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        # cv2.waitKey(0)
        # plt.imshow(img)
        # plt.show()
        # pass


if __name__ == '__main__':
    rospy.init_node('gs_viewpoint')
    vg = ViewpointGenerator()
    rospy.spin()