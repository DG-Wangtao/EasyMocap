import cv2
from detect_chessboard import create_chessboard
from easymocap.annotator import ImageFolder
import numpy as np
from os.path import join
import os
from tqdm import tqdm
from easymocap.annotator.file_utils import getFileList, read_json, save_json

## 竖向直线检测工具
# 1. 使用鼠标点选画出多边形，尽可能准确地包含要识别的区域
# 2. 程序会依次显示识别出的线段，按 空格键保存，ESC键退出这个图片，其它键跳过到下一条线
# 3. 标注结束后，按ESC退出，程序会显示出所有选中的线，按任意键继续
# 4. 对多张图片重复该过程
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param['stop']:
            return
        img = param['img']
        cache = img.copy()
        param['points'].append([x, y])
        pts = np.array([param['points']], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(cache, [pts], True, (0, 255, 255), 2)
        cv2.imshow('lines', cache)


# 灰度图转换
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Canny边缘检测
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


# 高斯滤波
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# 生成感兴趣区域即Mask掩模
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)  # 生成图像大小一致的zeros矩

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 填充函数
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    slope_min = .15  # 斜率低阈值
    slope_max = .95  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标
    max_y = image.shape[0]  # 最大y坐标

    results = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
            slope = fit[0]  # 斜率
            if slope_min < np.absolute(slope):
                results.append(line)

            # if slope_min < np.absolute(slope) <= slope_max:
            # results.append(line)

            # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
            # if slope > 0 and x1 > middle_x and x2 > middle_x:
            # right_y_set.append(y1)
            # right_y_set.append(y2)
            # right_x_set.append(x1)
            # right_x_set.append(x2)
            # right_slope_set.append(slope)

            # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
            # elif slope < 0 and x1 < middle_x and x2 < middle_x:
            # left_y_set.append(y1)
            # left_y_set.append(y2)
            # left_x_set.append(x1)
            # left_x_set.append(x2)
            # left_slope_set.append(slope)

    # 绘制左车道线
    # if left_y_set:
    #     lindex = left_y_set.index(min(left_y_set))  # 最高点
    #     left_x_top = left_x_set[lindex]
    #     left_y_top = left_y_set[lindex]
    #     lslope = np.median(left_slope_set)   # 计算平均值

    # 根据斜率计算车道线与图片下方交点作为起点
    # left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

    # 绘制线段
    # cv2.line(image, (left_x_bottom, max_y), (left_x_top, left_y_top), color, thickness)

    # 绘制右车道线
    # if right_y_set:
    #     rindex = right_y_set.index(min(right_y_set))  # 最高点
    #     right_x_top = right_x_set[rindex]
    #     right_y_top = right_y_set[rindex]
    #     rslope = np.median(right_slope_set)

    # 根据斜率计算车道线与图片下方交点作为起点
    # right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

    # 绘制线段
    # cv2.line(image, (right_x_top, right_y_top), (right_x_bottom, max_y), color, thickness)
    return results


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


# 把直线按第一个点 从左到右排序
def sort_lines(lines):
    return sorted(lines, key=(lambda x: [x[0][0], x[0][1]]), reverse=False)

# 把点从下往上排序
def sort_points(points):
    return sorted(points, key=(lambda x: x[1]), reverse=True)


def process_image(image):
    rho = 1  # 霍夫像素单位
    theta = np.pi / 180  # 霍夫角度移动步长
    hof_threshold = 20  # 霍夫平面累加阈值threshold
    min_line_len = 30  # 线段最小长度
    max_line_gap = 20  # 最大允许断裂长度

    kernel_size = 5  # 高斯滤波器大小size
    canny_low_threshold = 50  # canny边缘检测低阈值
    canny_high_threshold = canny_low_threshold * 3  # canny边缘检测高阈值

    # 手动划定直线检测区域
    cv2.namedWindow("lines", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("lines", cv2.WND_PROP_TOPMOST, cv2.WND_PROP_TOPMOST)
    mouse_param = {'img': image, 'stop': False, 'points': []}
    cv2.setMouseCallback("lines", on_click, mouse_param)
    cv2.imshow('lines', image)
    cv2.waitKey(0)
    mouse_param['stop'] = True

    # 灰度图转换
    gray = grayscale(image)

    # 高斯滤波
    blur_gray = gaussian_blur(gray, kernel_size)

    # Canny边缘检测
    edge_image = canny(blur_gray, canny_low_threshold, canny_high_threshold)

    # 生成Mask掩模
    # 只关注框选的部分
    vertices = np.array([
        [
            mouse_param['points']
        ]
    ], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)

    # 基于霍夫变换的直线检测
    lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(image)

    lines = sort_lines(lines)

    indexs = []
    stop = False
    while not stop:
        for i in range(len(lines)):
            cache = image.copy()

            # 显示已选中的线
            for j in indexs:
                line = lines[j]
                color = (0, (j * 100) % 255, 255 - (j * 100) % 255)
                cv2.line(cache, (line[0][0], line[0][1]),
                         (line[0][2], line[0][3]), color, 2, cv2.LINE_AA)

            color = (0, (i * 100) % 255, 255 - (i * 100) % 255)
            cv2.line(cache, (lines[i][0][0], lines[i][0][1]),
                     (lines[i][0][2], lines[i][0][3]), color, 2, cv2.LINE_AA)

            medium_x = int((lines[i][0][0] + lines[i][0][2]) / 2)
            medium_Y = int((lines[i][0][1] + lines[i][0][3]) / 2)
            cv2.putText(cache, str(i), (medium_x, medium_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.imshow('lines', cache)
            key = cv2.waitKey()

            if key == 32:
                # SPACE 保留该线段
                if i not in indexs:
                    indexs.append(i)
            elif key == 99:
                # 99
                if i in indexs:
                    indexs.remove(i)
            elif key == 27:
                # ESC next image
                stop = True
                break

    mouse_param['stop'] = True
    results = []
    cache = line_image.copy()
    for i in sorted(indexs):
        color = (0, (i * 100) % 255, 255 - (i * 100) % 255)

        cv2.line(image, (lines[i][0][0], lines[i][0][1]),
                 (lines[i][0][2], lines[i][0][3]), color, 2, cv2.LINE_AA)
        cv2.line(cache, (lines[i][0][0], lines[i][0][1]),
                 (lines[i][0][2], lines[i][0][3]), color, 2, cv2.LINE_AA)
        ## 显示line序号
        medium_x = int((lines[i][0][0] + lines[i][0][2]) / 2)
        medium_Y = int((lines[i][0][1] + lines[i][0][3]) / 2)
        cv2.putText(image, str(i), (medium_x, medium_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        x1 = float(lines[i][0][0])
        y1 = float(lines[i][0][1])
        x2 = float(lines[i][0][2])
        y2 = float(lines[i][0][3])

        arr = sort_points([[x1, y1], [x2, y2]])
        for ele in arr:
            results.append(ele)

    cv2.imshow('lines', image)
    cv2.waitKey()

    cv2.destroyWindow('lines')

    return results


def getChessboard3d(pattern, distance_x, distance_y, axis='xy'):
    object_points = np.zeros((pattern[1] * pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    object_points[:, :2] = object_points[:, :2]
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points[:, [0, 0]] = object_points[:, [0, 0]] * distance_y
    object_points[:, [1, 1]] = object_points[:, [1, 1]] * distance_x

    if axis == 'zx':
        object_points = object_points[:, [1, 2, 0]]
    return object_points


def create_chessboard(path, pattern, distance_x, distance_y, ext):
    print('Create chessboard {}'.format(pattern))
    keypoints3d = getChessboard3d(pattern, distance_x, distance_y)
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(path, ext=ext)
    keypoints3dlist = keypoints3d.astype(float).round(3).tolist()
    template = {
        'keypoints3d': keypoints3dlist,
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', 'chessboard').replace(ext, '.json')
        annname = join(path, annname)
        if os.path.exists(annname):
            # 覆盖keypoints3d
            data = read_json(annname)
            data['keypoints3d'] = template['keypoints3d']
            save_json(annname, data)
        else:
            save_json(annname, template)


def detect_line_item(path, out, imgname, annotname):
    img = cv2.imread(imgname)
    annots = read_json(annotname)
    points = process_image(img)
    for i in range(len(points)):
        annots['keypoints2d'][i] = [points[i][0], points[i][1], 1.0]

    save_json(annotname, annots)
    if points is None:
        if args.debug:
            print('Cannot find {}'.format(imgname))
        return
    outname = join(out, imgname.replace(path + '/images/', ''))
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    cv2.imwrite(outname, img)


def detect_line(path, out, pattern, distance, length):
    create_chessboard(path, pattern, distance, length, ext=args.ext)
    dataset = ImageFolder(path, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    if args.silent:
        trange = range(len(dataset))
    else:
        trange = tqdm(range(len(dataset)))

    for i in trange:
        imgname, annotname = dataset[i]
        detect_line_item(path, out, imgname, annotname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--len', type=float, help='每条线的实际长度')
    parser.add_argument('--distance', type=float, help='两条直线间的实际距离')
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
                        help='The pattern of the lines: (row, col). (5, 2) for 5 lines with 2 points ', default=(5, 2))
    args = parser.parse_args()
    detect_line(args.path, args.out, args.pattern, args.distance, args.len)
