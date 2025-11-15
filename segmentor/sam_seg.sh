
###
 # @Author: EasonZhang
 # @Date: 2024-07-23 10:44:13
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2025-11-15 15:12:41
 # @FilePath: /SA2M/hydra-mesa/segmentor/sam_seg.sh
 # @Description: TBD
 # 
 # Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
### 

# cuda_id=0
# img_path=/opt/data/private/SA2M/hydra-mesa/SAM/demo/src/assets/data/dogs.jpg
# save_folder=/opt/data/private/SA2M/hydra-mesa/segmentor/seg_res
# save_name=seg_res_dogs.jpg

# CUDA_VISIBLE_DEVICES=$cuda_id python ImgSAMSeg.py --img_path $img_path --save_folder $save_folder --save_name $save_name --sam_name SAM


# for sam embedding visualization
# cuda_id=0
# img_path0=/opt/data/private/SA2M/data/scannet_test_1500/scene0707_00/color/15.jpg
# img_path1=/opt/data/private/SA2M/data/scannet_test_1500/scene0707_00/color/45.jpg
# save_folder=/opt/data/private/A2PM-git/A2PM-MESA/R1/res/seg_res
# save_name0=seg_res_15.jpg
# save_name1=seg_res_45.jpg

# CUDA_VISIBLE_DEVICES=$cuda_id python ImgSAMSeg.py --img_path $img_path0 --save_folder $save_folder --save_name $save_name0 --sam_name SAM --embed_name img_emb_15
# CUDA_VISIBLE_DEVICES=$cuda_id python ImgSAMSeg.py --img_path $img_path1 --save_folder $save_folder --save_name $save_name1 --sam_name SAM --embed_name img_emb_45


# for common use
img_folder=""
img_posfix=".png"
cuda_id=0
save_folder=""


for img in `ls $img_folder/*$img_posfix`; do
    img_name=$(basename $img)
    # get rid of .png
    img_name=${img_name%.*}
    echo seg $img_name
    CUDA_VISIBLE_DEVICES=$cuda_id python ImgSAMSeg.py --img_path $img --save_folder $save_folder --save_name $img_name --sam_name SAM --embed_name ${img_name}_emb
done