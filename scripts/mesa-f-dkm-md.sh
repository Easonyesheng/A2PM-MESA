
###
 # @Author: EasonZhang
 # @Date: 2024-07-18 15:36:50
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2024-07-29 10:43:38
 # @FilePath: /SA2M/hydra-mesa/scripts/qua-res-generator-dmesa-dkm-md.sh
 # @Description: TBD
 # 
 # Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
### 
dataset=MegaDepth
cuda_id=1
project_name=mesa-f-dkm-md-eval
exp_root_path=/opt/data/private/A2PM-git/A2PM-MESA

already_done_name_file_folder=${exp_root_path}/res/${project_name}-res/ratios
already_done_name_file=$(ls ${already_done_name_file_folder}/*pose_err_names.txt | head -n 1)

pair_txt=${exp_root_path}/scripts/megadepth_1500_pairs.txt
scene_name=MegaDepth

# get the scene name and pair from pair_txt
# the format of pair_txt is: scene_name_pair0_pair1
while read line
do
    echo "line: ${line}"
    # parse line to pair0 and pair1, line is {pair0} {pair1}
    pair0=$(echo ${line} | awk '{print $1}')
    pair1=$(echo ${line} | awk '{print $2}')

    echo "pair0: ${pair0}"
    echo "pair1: ${pair1}"
    
    # # get the last part of the pair1, which is separated by _
    # pair1_last=$(echo ${pair1} | awk -F_ '{print $NF}') 

    complete_pair_name=MegaDepth_${pair0}_${pair1} 
    echo "complete_pair_name: ${complete_pair_name}"

    # if $scene_$pair0_$pair1 in already_done_name_file, continue
    if [ -f "${already_done_name_file}" ];then
        if grep -q "${complete_pair_name}" ${already_done_name_file};then
            echo ${complete_pair_name} already done
            continue
        fi
    fi

    echo "performing test on ${complete_pair_name}"

    CUDA_VISIBLE_DEVICES=$cuda_id python test_a2pm.py \
        +experiment=a2pm_mesa_egam_dkm_megadepth \
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



