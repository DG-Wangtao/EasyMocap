'''
  @ Date: 2021-05-26 13:19:12
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-17 15:21:46
  @ FilePath: /EasyMocap/apps/annotation/annot_calib.py
'''
import cv2

from apps.calibration.calib_extri import calib_extri
from apps.calibration.check_calib import load_cube
from easymocap.annotator.basic_visualize import plot_text, resize_to_screen, vis_line
from easymocap.mytools.vis_base import plot_point, plot_points2d
from easymocap.annotator import ImageFolder
from easymocap.annotator import vis_point
from easymocap.annotator import AnnotBase
from easymocap.annotator.file_utils import getFileList, save_annot
from easymocap.mytools import read_json, save_json, projectN3, Undistort, read_camera
from easymocap.mytools import plot_cross, plot_line, get_rgb
import numpy as np
from tqdm import tqdm
from os.path import join
import os
from easymocap.annotator.chessboard import get_lines_chessboard
from easymocap.annotator.vanish_callback import calc_vanishpoint

class Matcher:
    def __init__(self, path, mode, args) -> None:
        if mode == 'chessboard':
            pattern = args.pattern
            lines, lines_color = get_lines_chessboard((pattern[0], pattern[1]))
            self.nJoints = pattern[0]*pattern[1]
        else:
            annots = read_json(join(path, 'calib.json'))
            lines = annots['lines']
            lines_color = annots['lines_color']
            keypoints3d = np.array(annots['keypoints3d'])
            create_chessboard(path, keypoints3d, out=args.annot)
            self.nJoints = len(keypoints3d)
        self.lines = lines
        self.lines_color = lines_color
        self.cache_lines = []
        self.cnt = -1
        self.hint()
        self.args = args
        self.save_callback = None
    def set_callback(self, callback):
        self.save_callback = callback
    
    def hint(self):
        self.cnt = (self.cnt + 1)%self.nJoints
        print('>>> label point {}'.format(self.cnt))
        
    def add(self, annotator, param, conf=1.):
        "add matched points"
        click = param['click']
        if click is not None:
            param['annots']['keypoints2d'][self.cnt] = [click[0], click[1], conf]
            param['annots']['visited'] = True
            param['click'] = None
            if args.cube:
                self.view_cube(param)

        self.hint()
    def move(self, annotator, param, conf=1.):
        key = param['key']
        point = param['annots']['keypoints2d'][self.cnt]
        # 1 down
        if key == 1:
            point[1] = point[1] + 1
        # 0 up
        if key == 0:
            point[1] = point[1] - 1
        # 2 left
        if key == 2:
            point[0] = point[0] - 1
        # 3 right
        if key == 3:
            point[0] = point[0] + 1
        param['annots']['keypoints2d'][self.cnt] = point
        param['annots']['visited'] = True
        param['click'] = None

        if args.cube:
            self.view_cube(param)

    def view_cube(self, param):

        # 保存改动
        save_annot(param['annname'], param['annots'])
        self.save_callback()
        # 计算外参
        calib_extri(self.args.path, self.args.image, intriname=self.args.intri, image_id=0, ext=args.ext)
        # 显示正方体
        points, lines = load_cube()
        imgname = param['imgname']
        paths = imgname.split(os.sep)
        camera = paths[len(paths) - 2]
        check_sene(camera, self.args.path, self.args.out, points, lines)


    def add_conf(self, annotator, param):
        self.add(annotator, param, conf=0.5)
    
    def add_point_by_2lines(self, annotator, param):
        start = param['start']
        end = param['end']
        if start is None:
            return 0
        if len(self.cache_lines) < 2:
            self.cache_lines.append((start, end))
            param['start'] = None
            param['end'] = None
        if len(self.cache_lines) == 2:
            # calculate intersect
            inp = np.zeros((2, 2, 3)) # 2, points, (x, y, c)
            for i in range(len(self.cache_lines)):
                start, end = self.cache_lines[i]
                inp[0, i, 0] = start[0]
                inp[0, i, 1] = start[1]
                inp[0, i, 2] = 1.
                inp[1, i, 0] = end[0]
                inp[1, i, 1] = end[1]
                inp[1, i, 2] = 1.
            intersect = calc_vanishpoint(inp)
            click = (int(intersect[0]), int(intersect[1]))
            param['click'] = click
            self.cache_lines = []
    
    def clear(self, annotator, param):
        "clear all points"
        for i in range(self.nJoints):
            param['annots']['keypoints2d'][i][2] = 0.
        self.cnt = self.nJoints - 1
        self.hint()
    
    def clear_point(self, annotator, param):
        "clear current points"
        param['annots']['keypoints2d'][self.cnt][2] = 0.

    def vis(self, img, annots, **kwargs):
        lw = max(int(round(img.shape[0]/500)), 1)
        width = lw * 5
        k2d = np.array(annots['keypoints2d'])
        # cache lines
        for nl, (start, end) in enumerate(self.cache_lines):
            plot_line(img, start, end, lw, (255, 255, 200))
            
        for nl, (i, j) in enumerate(self.lines):
            if k2d[i][2] > 0 and k2d[j][2] > 0:
                plot_line(img, k2d[i], k2d[j], lw, self.lines_color[nl])
        for i, (x, y, c) in enumerate(k2d):
            if c > 0:
                plot_cross(img, x, y, self.lines_color[min(len(self.lines)-1, i+1)], width=width, lw=lw)
                plot_point(img, x, y, r=lw*2, col=self.lines_color[min(len(self.lines)-1, i+1)], pid=i)
            if i == self.cnt:
                plot_point(img, x, y, r=lw*16, col=(127, 127, 255), pid=-1, circle_type=lw*2)
        return img

    def print(self, annotator, **kwargs):
        print(self.annots)

def check_sene(cam, path, out, points3d, lines):
    cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
    cameras.pop('basenames')
    camera = cameras[cam]
    imgname = join(path, 'images', cam, '{:06d}.jpg'.format(0))
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    img = Undistort.image(img, camera['K'], camera['dist'])
    kpts_repro = projectN3(points3d, camera['P'][None, :, :])[0]
    plot_points2d(img, kpts_repro, lines, lw=1, putText=True)
    cv2.imshow('vis', img)
    cv2.waitKey(0)


def create_chessboard(path, keypoints3d, out='annots'):
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(path, ext='.jpg')
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', out).replace('.jpg', '.json')
        annname = join(path, annname)
        if not os.path.exists(annname):
            save_json(annname, template)
        elif True:
            annots = read_json(annname)
            annots['keypoints3d'] = template['keypoints3d']
            save_json(annname, annots)

def annot_example(path, args):
    # define datasets
    calib = Matcher(path, mode=args.mode, args=args)
    dataset = ImageFolder(path, annot=args.annot)
    # define visualize
    vis_funcs = [vis_point, vis_line, calib.vis, resize_to_screen, plot_text]
    key_funcs = {
        ' ': calib.add,
        'z': calib.add_conf,
        'p': calib.add_point_by_2lines,
        'c': calib.clear_point,
        'C': calib.clear,
        0: calib.move,
        1: calib.move,
        2: calib.move,
        3: calib.move,
    }
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset,
        key_funcs=key_funcs,
        vis_funcs=vis_funcs)
    calib.set_callback(annotator.save)
    while annotator.isOpen:
        annotator.run()


if __name__ == "__main__":
    from easymocap.annotator import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(9, 6))
    parser.add_argument('--mode', type=str, default='chessboard')
    # parser.add_argument('--vis', action='store_true')
    # parser.add_argument('--out', type=str, help='with camera parameters')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--cube', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--intri', type=str, default=None)
    args = parse_parser(parser)

    annot_example(args.path, args)