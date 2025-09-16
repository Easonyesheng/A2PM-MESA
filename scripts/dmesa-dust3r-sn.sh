
###
 # @Author: EasonZhang
 # @Date: 2024-07-18 14:38:31
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2025-09-12 10:32:37
 # @FilePath: /SA2M/hydra-mesa/scripts/qua-res-generator-dmesa-dkm-sn.sh
 # @Description: TBD
 # 
 # Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
### 
dataset=ScanNet
cuda_id=0
project_name=dmesa-dust3r-sn-eval
exp_root_path=/opt/data/private/A2PM-git/A2PM-MESA

already_done_name_file_folder=${exp_root_path}/res/${project_name}-res/ratios
already_done_name_file=$(ls ${already_done_name_file_folder}/*pose_err_names.txt | head -n 1)

pair_txt=${exp_root_path}/scripts/scannet_pairs.txt

# get the scene name and pair from pair_txt
# the format of pair_txt is: scene_name_pair0_pair1
while read line
do
    # split line by _
    arr=(${line//_/ })
    echo scene_name = ${arr[0]}_${arr[1]}
    echo pair0 = ${arr[2]}
    echo pair1 = ${arr[3]}
    scene_name=${arr[0]}_${arr[1]}
    pair0=${arr[2]}
    pair1=${arr[3]}
    
    # if $scene_name_$pair0_$pair1 in already_done_name_file, continue
    if [ -f "${already_done_name_file}" ];then
        if grep -q "${scene_name}_${pair0}_${pair1}" ${already_done_name_file};then
            echo ${scene_name}_${pair0}_${pair1} already done
            continue
        fi
    fi

    CUDA_VISIBLE_DEVICES=$cuda_id python test_a2pm.py \
        +experiment=a2pm_dmesa_egam_dust3r_scannet \
        test_area_acc=False \
        test_pm_acc=False \
        verbose=0 \
        name=${project_name} \
        dataset_name=$dataset \
        dataset.scene_name=$scene_name \
        dataset.image_name0=$pair0 \
        dataset.image_name1=$pair1
# break
done < $pair_txt



