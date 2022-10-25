#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/$(pwd)

## extract images from videos
echo "===================="
echo "extracting images from videos"
echo "===================="
#rm -r $(pwd)/data/intri/images
#python3 scripts/preprocess/extract_video.py $(pwd)/data/intri --no2d --step 200 --end 10000000

echo "===================="
echo "detecting chessboard for intri parameter"
echo "===================="
#rm -r $(pwd)/data/intri/chessboard
#rm -r $(pwd)/data/intri/output
#python3 apps/calibration/detect_chessboard.py $(pwd)/data/intri --out $(pwd)/data/intri/output/calibration --pattern 8,7 --grid 0.1

# 调整内参标记
#python3 apps/annotation/annot_calib.py $(pwd)/data/intri --mode chessboard --annot chessboard --pattern 8,7


echo "===================="
echo "Intrinsic Parameter Calibration"
echo "===================="
#python3 apps/calibration/calib_intri.py $(pwd)/data/intri

## 直线检测
## 1. 通过鼠标点击选择一个ROI区域
## 2. 程序依次显示在该区域中检测出的直线
#     按 空格键选中，按C键取消选中，按N键显示下一条直线，按ESC键退出选择（会显示出当前选择的所有线条及编号，按空格确认）
#rm -r $(pwd)/data/extri/chessboard
#rm -r $(pwd)/data/extri/output
#python3 apps/calibration/detect_line.py  $(pwd)/data/extri --out $(pwd)/data/extri/output/calibration --pattern 5,3 --distance 1 --len 1.25

## 手动处理地面标记数据
## 在原有的快捷键基础上，添加以下操作：
## - 上下左右 像素级移动当前点位置，并在另一个界面实时显示外参计算结果及立方体
#python3 apps/annotation/annot_calib.py $(pwd)/data/extri --mode chessboard --annot chessboard --pattern 5,3 --cube --show --grid --out $(pwd)/data/extri/ --intri $(pwd)/data/intri/output/intri.yml


## 上面实时计算了外参和立方体显示，下面的逻辑其实不需要了
echo "===================="
echo "Extrinsic Parameter Calibration"
echo "===================="
#rm -r $(pwd)/data/extri/extri.yml
#rm -r $(pwd)/data/extri/intri.yml
#python3 apps/calibration/calib_extri.py $(pwd)/data/extri
#python3 apps/calibration/calib_extri.py $(pwd)/data/extri --intri $(pwd)/data/intri/output/intri.yml


## Check the calibration results with chessboard
#python3 apps/calibration/check_calib.py $(pwd)/data/extri --out $(pwd)/data/extri --vis --show


# Check the results with a cube
#python3 apps/calibration/check_calib.py $(pwd)/data/extri --out $(pwd)/data/extri --cube --show --grid