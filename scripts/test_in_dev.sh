###
 # @Author: EasonZhang
 # @Date: 2024-06-17 22:40:17
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2024-07-27 15:35:46
 # @FilePath: /SA2M/hydra-mesa/scripts/test_in_dev.sh
 # @Description: TBD
 # 
 # Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
### 

# ScanNet
dataset=ScanNet
scene=scene0720_00
pair0=180
pair1=2580

# MESA+DKM+ScanNet - tested
python test_a2pm.py \
    +experiment=a2pm_mesa_egam_dkm_scannet \
    name=test \
    dataset_name=$dataset \
    dataset.scene_name=$scene \
    dataset.image_name0=$pair0 \
    dataset.image_name1=$pair1 \

# # DMESA+DKM+ScanNet - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_dkm_scannet \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# DMESA+SPSG+ScanNet - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_spsg_scannet \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \
    
# MESA+SPSG+ScanNet - tested
# python test_a2pm.py \
#     +experiment=a2pm_mesa_egam_spsg_scannet \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# MESA+loftr+ScanNet - tested
# python test_a2pm.py \
#     +experiment=a2pm_mesa_egam_loftr_scannet \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# DMESA+LoFTR+ScanNet - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_loftr_scannet \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

############################################
# MegaDepth
dataset=MegaDepth
scene=md # no use
pair0='0022_0.1_0.3_1401'
pair1='0022_0.1_0.3_810'

# # MESA+DKM+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_mesa_egam_dkm_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# # DMESA+DKM+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_dkm_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \


# DMESA+SPSG+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_spsg_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# MESA+SPSG+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_mesa_egam_spsg_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# DMESA+LoFTR+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_dmesa_egam_loftr_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \

# # MESA+LoFTR+MegaDepth - tested
# python test_a2pm.py \
#     +experiment=a2pm_mesa_egam_loftr_megadepth \
#     name=test \
#     dataset_name=$dataset \
#     dataset.scene_name=$scene \
#     dataset.image_name0=$pair0 \
#     dataset.image_name1=$pair1 \