
###
 # @Author: EasonZhang
 # @Date: 2024-07-23 10:44:13
 # @LastEditors: EasonZhang
 # @LastEditTime: 2024-07-23 10:50:31
 # @FilePath: /SA2M/hydra-mesa/segmentor/sam_seg.sh
 # @Description: TBD
 # 
 # Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
### 

cuda_id=0
img_path=/opt/data/private/SA2M/hydra-mesa/SAM/demo/src/assets/data/dogs.jpg
save_folder=/opt/data/private/SA2M/hydra-mesa/segmentor/seg_res
save_name=seg_res_dogs.jpg

CUDA_VISIBLE_DEVICES=$cuda_id python ImgSAMSeg.py --img_path $img_path --save_folder $save_folder --save_name $save_name