#!/bin/bash

input=$1

# 跳过表头
for line in `tail -n +2 $input`
do
  # 按','分割字符串到数组中
  OIFS=$IFS
  IFS=,
  col=($line)

  path=${col[0]}
  start=${col[1]}
  end=${col[2]}
  echo "processing $path from $start to $end"
  # 输出到同级目录下 原文件_开始时间.MP4
  output="${path/.MP4/_$start.MP4}"
  output=$(echo $output | sed  's/:/-/g')
  echo "writing to "$output
  IFS=$OIFS

  # 不能再 -i 前加 -ss参数进行查找，因为慢放导致时间不对
  ffmpeg -loglevel warning  -i $path -ss $start -to $end -vcodec copy -acodec copy -strict experimental -y $output
done


# 加速8倍 fps 29.97 -> 239.76
# ffmpeg -loglevel warning -i cam1.MP4 -r 239.76  -filter:v "setpts=0.125*PTS" speedup.MP4

# 减速8倍 fps 239.76 -> 29.97
# ffmpeg  -loglevel warning -i camg0.MP4 -r 29.97  -filter:v "setpts=8*PTS" slowdown.MP4

# 提取视频的时间戳
# ffmpeg -hide_banner -i cam1.MP4 -filter:v showinfo -f null /dev/null

# 按帧率计算应有的时间间隔
#1/29.97=0.03336670
#1/30 = 0.033333

# 根据上面的命令，得到某两帧间隔时间
# n:1420 - n:1419
#4264260-4261257 = 3,003
#47.3807-47.3473 =  0.0334

# n: 100 pts: 300300 pts_time:3.33667
# n: 200 pts: 600600 pts_time:6.67333
# n: 1100 pts:3303300 pts_time:36.7033

# 6.67333 - 3.33667 = 3.33666
# 3.33666 / 100 = 0.033366
# 36.7033 - 3.33667= 33.36663
# 33.36663 / 1000 = 0.03336663
