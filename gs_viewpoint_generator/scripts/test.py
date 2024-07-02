# /usr/bin/python3
import cv2
import numpy as np
from math import hypot

# img是地图，exp是探测范围，vis是可视化
# known_img是已知地图
# img: 255是obstacle，0是free
# exp: 255是已知区域，0是未知区域
# known_img: 255是free，0是obstacle
# vis: 0是obstacle，255是free，100是未知区域

img = np.zeros((1000, 1000), np.uint8)
exp = np.zeros((1000, 1000), np.uint8)
known_img = np.zeros((1000, 1000), np.uint8)
vis = np.ones((1000, 1000, 3), np.uint8)
line = np.zeros((1000, 1000), np.uint8)
line_tmp = np.zeros((1000, 1000), np.uint8)

vis = vis * 100

# exp右上角是已知区域
for i in range(1000):
    exp[i, i:] = 255
exp[:100] = 0
exp[:, 900:] = 0
exp[310:, :500] = 0

# img[500:700, 100:400] = 255
# img[100:400, 300:400] = 255
# img[300:400, 700:800] = 255
# 四面围墙
img[100:150, 100:900] = 255
img[100:900, 100:150] = 255
img[850:900, 100:900] = 255
img[100:900, 850:900] = 255
# 中间的墙
img[300:900, 490:510] = 255
img[200:250, 700:900] = 255
img[400:440, 510:650] = 255
img[300:350, 300:490] = 255
img[600:700, 100:300] = 255

vis_img = img.copy()
vis_img[vis_img==255] = 100
vis_img[vis_img==0] = 255
vis_img[vis_img==100] = 0
cv2.imshow('地图', vis_img)

# # 路径点
# path_history = []
# path_history.append([700, 600])

# 选取轮廓
mask = (img == 255) & (exp == 255)
known_img[mask] = 255
cv2.imshow('已知地图', known_img)
# 腐蚀
kernel = np.ones((100, 100), np.uint8)
# 膨胀
dilation = cv2.dilate(known_img, kernel, iterations=1)
# known_img = known_img * 255
mask = (dilation == 255) & (exp == 255)
# known_img[dilation == 255] = 255

# 寻找轮廓
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
cv2.drawContours(line_tmp, contours, -1, 255, 3)
mask = (line_tmp == 255) & (exp == 255)
line[mask] = 255
cv2.imshow('view line', line)

# 可视化
vis[img==255] = [0, 0, 0]
mask = (img == 0) & (exp == 255)
vis[mask] = [255, 255, 255]
vis[line==255] = [255, 0, 0]

points = []
def get_list(point1, point2):
    global points
    x1, y1 = point1
    x2, y2 = point2
    dif_x = x2 - x1
    dif_y = y2 - y1
    dif = hypot(dif_x, dif_y)
    num = dif // 40
    for i in range(int(num)):
        x = x1 + dif_x / num * i
        y = y1 + dif_y / num * i
        points.append([int(x), int(y)])
for contour in contours:
    for num in range(len(contour)):
        if contour[num][0][0] < 100 or contour[num][0][1] < 100:
            continue
        if contour[num][0][0] > 900 or contour[num][0][1] > 900:
            continue
        if num + 1 < len(contour):
            get_list(contour[num][0], contour[num+1][0])
        else:
            get_list(contour[num][0], contour[0][0])

tsp_points = []
view_points = []
for point in points:
    if exp[point[1], point[0]] == 255 and img[point[1], point[0]] == 0:
        cv2.circle(vis, (point[0], point[1]), 5, (0, 0, 255), -1)
        tsp_points.append(point)
        view_points.append(point)

# 计算视场
view_left = []
view_right = []
FOV = 70/180*np.pi
view_length = 100*np.tan(FOV/2)
colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
for i in range(len(view_points)):
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
    img_tmp1 = np.zeros((1000, 1000), np.uint8)
    img_tmp2 = np.zeros((1000, 1000), np.uint8)
    cv2.fillPoly(img_tmp1, [np.array([view_points[i], view_point_tmp1l, view_point_tmp1r])], 255)
    cv2.fillPoly(img_tmp2, [np.array([view_points[i], view_point_tmp2l, view_point_tmp2r])], 255)
    if np.sum(img_tmp1 & known_img) < np.sum(img_tmp2 & known_img):
        yaw += np.pi
    
    view_left.append([int(view_points[i][0]+view_length*np.cos(yaw+FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw+FOV/2))])
    view_right.append([int(view_points[i][0]+view_length*np.cos(yaw-FOV/2)), int(view_points[i][1]+view_length*np.sin(yaw-FOV/2))])
    cv2.line(vis, (view_points[i][0], view_points[i][1]), (view_left[i][0], view_left[i][1]), colors[i%3], 1)
    cv2.line(vis, (view_points[i][0], view_points[i][1]), (view_right[i][0], view_right[i][1]), colors[i%3], 1)
    # 填充颜色，透明
    vis_tmp = vis.copy()
    cv2.fillPoly(vis_tmp, [np.array([view_points[i], view_left[i], view_right[i]])], colors[i%3], )
    vis = cv2.addWeighted(vis, 0.5, vis_tmp, 0.5, 0)

# 起始点
cv2.circle(vis, (700, 600), 5, (0, 0, 0), -1)
tsp_points.append([700, 600])

def frontier(i, j):
    for point in tsp_points:
        if hypot(i-point[1], j-point[0]) < 40:
            return False
    if exp[i-1, j] == 0:
        return True
    if exp[i+1, j] == 0:
        return True
    if exp[i, j-1] == 0:
        return True
    if exp[i, j+1] == 0:
        return True
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if exp[i, j] == 255 and img[i, j] == 0 and frontier(i, j):
            cv2.circle(vis, (i, j), 7, (255, 255, 0), -1)
            # tsp_points.append([i, j])

import os
import shutil
import math
import numpy as np

def clac_distance(X, Y):
    """
    计算两个城市之间的欧氏距离，二范数
    :param X: 城市X的坐标.np.array数组
    :param Y: 城市Y的坐标.np.array数组
    :return:
    """
    distance_matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            if i == j:
                continue

            distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distance_matrix[i][j] = distance

    return distance_matrix

#读取31座城市坐标
coord = []
# with open("data.txt", "r") as lines:
#     lines = lines.readlines()
for line in tsp_points:
    coord.append(line)
coord = np.array(coord)
w, h = coord.shape
coordinates = np.zeros((w, h), float)
for i in range(w):
    for j in range(h):
        coordinates[i, j] = float(coord[i, j])
city_x=coordinates[:,0]
city_y=coordinates[:,1]
#城市数量
city_num = coordinates.shape[0]
CostMatrix = clac_distance(city_x, city_y)*1000    #将距离矩阵放大1000倍（LKH算法只能处理整数）

fname_tsp = "city031"
user_comment = "a comment by the user"

# Change these directories based on where you have 
# a compiled executable of the LKH TSP Solver
lkh_dir = '/LKH-2.0.10/'
tsplib_dir = '/TSPLIB/'
lkh_cmd = 'LKH'
pwd=os.getcwd()


def writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment):

    dims_tsp = len(CostMatrix)
    name_line = 'NAME : ' + fname_tsp + '\n'
    type_line = 'TYPE: TSP' + '\n'
    comment_line = 'COMMENT : ' + user_comment + '\n'
    tsp_line = 'TYPE : ' + 'TSP' + '\n'
    dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
    edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n' # explicit only
    edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
    display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n' # 'NO_DISPLAY'
    edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
    eof_line = 'EOF\n'
    Cost_Matrix_STRline = []
    for i in range(0,dims_tsp):
        cost_matrix_strline = ''
        for j in range(0,dims_tsp-1):
            cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j])) + ' '

        j = dims_tsp-1
        cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j]))
        cost_matrix_strline = cost_matrix_strline + '\n'
        Cost_Matrix_STRline.append(cost_matrix_strline)
    
    fileID = open((pwd + tsplib_dir + fname_tsp + '.tsp'), "w")
    print(name_line)
    fileID.write(name_line)
    fileID.write(comment_line)
    fileID.write(tsp_line)
    fileID.write(dimension_line)
    fileID.write(edge_weight_type_line)
    fileID.write(edge_weight_format_line)
    fileID.write(edge_weight_section_line)
    for i in range(0,len(Cost_Matrix_STRline)):
        fileID.write(Cost_Matrix_STRline[i])

    fileID.write(eof_line)
    fileID.close()

    fileID2 = open((pwd + tsplib_dir + fname_tsp + '.par'), "w")

    problem_file_line = 'PROBLEM_FILE = ' + pwd + tsplib_dir + fname_tsp + '.tsp' + '\n' # remove pwd + tsplib_dir
    optimum_line = 'OPTIMUM 378032' + '\n'
    move_type_line = 'MOVE_TYPE = 5' + '\n'
    patching_c_line = 'PATCHING_C = 3' + '\n'
    patching_a_line = 'PATCHING_A = 2' + '\n'
    runs_line = 'RUNS = 10' + '\n'
    tour_file_line = 'TOUR_FILE = ' + fname_tsp + '.txt' + '\n'

    fileID2.write(problem_file_line)
    fileID2.write(optimum_line)
    fileID2.write(move_type_line)
    fileID2.write(patching_c_line)
    fileID2.write(patching_a_line)
    fileID2.write(runs_line)
    fileID2.write(tour_file_line)
    fileID2.close()
    return fileID, fileID2

def copy_toTSPLIBdir_cmd(fname_basis):
    srcfile=pwd + '/' + fname_basis + '.txt'
    dstpath=pwd + tsplib_dir
    shutil.copy(srcfile, dstpath)

def run_LKHsolver_cmd(fname_basis):
    run_lkh_cmd =  pwd + lkh_dir  + lkh_cmd + ' ' + pwd + tsplib_dir + fname_basis + '.par'
    os.system(run_lkh_cmd)

def rm_solution_file_cmd(fname_basis):
    del_file=pwd + '/' + fname_basis + '.txt'
    os.remove(del_file)

def main(): 
    
    [fileID1,fileID2] = writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment)
    run_LKHsolver_cmd(fname_tsp)
    copy_toTSPLIBdir_cmd(fname_tsp)
    rm_solution_file_cmd(fname_tsp)
    

if __name__ == "__main__":
    main()

with open('TSPLIB/city031.txt', 'r') as f:
    lines = f.readlines()
    lines = lines[6:]
    lines = lines[:-2]
# print(city_x[1])
for line in range(len(lines)-1):
    # line = int(lines[line])
    # print(lines[line], lines[line+1])
    cv2.line(vis, (int(city_x[int(lines[line])-1]), int(city_y[int(lines[line])-1])), (int(city_x[int(lines[line+1])-1]), int(city_y[int(lines[line+1])-1])), (0, 255, 0), 2)
    # 添加编号
    cv2.putText(vis, str(line), (int(city_x[line]), int(city_y[line])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.line(vis, (int(city_x[int(lines[-1])-1]), int(city_y[int(lines[-1])-1])), (int(city_x[int(lines[0])-1]), int(city_y[int(lines[0])-1])), (0, 255, 0), 2)

cv2.imshow('可视化', vis)
cv2.waitKey(0)