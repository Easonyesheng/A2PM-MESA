##                                           Windows下运行A2PM项目

## 一 进行Segmentation Preprocessing

**1.**
下载[SAM](https://github.com/facebookresearch/segment-anything)粘贴到A2PM-MESA目录下；

   Download [SAM](https://github.com/facebookresearch/segment-anything) and paste it into the `A2PM-MESA` directory.

**2.**
下载[SAM2](https://github.com/facebookresearch/sam2)，将其中的sam2粘贴到A2PM-MESA\segmentor目录下（别问为什么，SAM2项目路径内部问题，源码内部有整段解释为何会有包错误）；

Download [SAM2](https://github.com/facebookresearch/sam2) and paste the `sam2` in it into the `A2PM-MESA\segmentor` directory (don't ask why, there is a problem inside the SAM2 project path, there is a whole paragraph inside the source code explaining why there is a package error);

**3.**
修改`SAMSeger.py`文件内         
modify file `SAMSeger.py`  

```
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from SAM2.sam2.build_sam import build_sam2
from SAM2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
```

改为    
to 

```
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
```

**4.**
下载`sam_vit_h_4b8939.pth`模型文件放在`A2PM-MESA\segmentor`目录下，同时修改该目录下的`ImgSAMSeg.py`文件，将`"sam_model_path": f"{current_path}/SAM/sam_vit_h_4b8939.pth",`替换为你下载的模型地址

  Download the `sam_vit_h_4b8939.pth` model file and place it in the `A2PM-MESA\segmentor` directory. Additionally, modify the `ImgSAMSeg.py` file in the same directory.

```python
SAM_configs = {
    "SAM_name": "SAM",
    "W": 640,
    "H": 480,
    "sam_model_type": "vit_h",
    "sam_model_path": f"{current_path}/SAM/sam_vit_h_4b8939.pth",
    "save_folder": "",
    "points_per_side": 16,
}
```

Replace with the address of the model you downloaded。

**5.**

```bash
cd F:\VSCodeProject\A2PM-MESA\segmentor
```
(modify to your address)

```python
python ImgSAMSeg.py --img_path "F:\VSCodeProject\A2PM-MESA\dataset\scannet_test_1500\scene0720_00\color\180.jpg" --save_folder "F:\VSCodeProject\A2PM-MESA\result\private\SA2M\data\SAMRes\scene0720_00" --save_name "180"
```

```python
python ImgSAMSeg.py --img_path "F:\VSCodeProject\A2PM-MESA\dataset\scannet_test_1500\scene0720_00\color\300.jpg" --save_folder "F:\VSCodeProject\A2PM-MESA\result\private\SA2M\data\SAMRes\scene0720_00" --save_name "300"
```

结果的地址要与 `A2PM-MESA\conf\dataset\scannet_sam.yaml` 文件下的路径一致

correspond with the address in `A2PM-MESA\conf\dataset\scannet_sam.yaml`

## 二 根据第一步的结果进行MESA+DKM拼配

## MESA+DKM matching according to the results of the first step

```bash
cd F:\VSCodeProject\A2PM-MESA\scripts

python test_a2pm.py  +experiment=a2pm_mesa_egam_dkm_scannet
```

## 三 进行基准测试  Perform benchmark testing.

**1.**
根据`A2PM-MESA\scripts\scannet_pairs.txt`中需要的图片依次调用第一步中的命令生成所需`.npy`文件（只进行部分测试,前十几张照片即可）

   According to the pictures required in `A2PM-MESA\scripts\scannet_pairs.txt`, call the command of the first step to generate the required `.npy` file (only part of the test, the first dozen photos can be used).

**2.**
在`A2PM-MESA\scripts`下新建`mesa-f-dkm-sn.py`文件,文件第8，10，11行需要自行修改

  (Create a `mesa-f-dkm-sn.py` file under `A2PM-MESA\scripts`, and you need to modify lines 8, 10, and 11 of the file)

```
import os
import subprocess

# 设置参数
dataset = "ScanNet"
cuda_id = 0
project_name = "mesa-f-egam-dkm-sn-eval-res"
exp_root_path = "F:\\VSCodeProject\\A2PM-MESA"

already_done_name_file_folder = os.path.join(exp_root_path, "result", "private","A2PM-MESA","res", f"{project_name}", "ratios")
pair_txt = os.path.join(exp_root_path, "scripts", "scannet_pairs.txt")

# 获取已经完成的文件
already_done_name_file = None
for file in os.listdir(already_done_name_file_folder):
    if file.endswith("pose_err_names.txt"):
        already_done_name_file = os.path.join(already_done_name_file_folder, file)
        break

# 读取对文件
with open(pair_txt, 'r') as f:
    for line in f:
        # 解析每一行
        arr = line.strip().split('_')
        scene_name = f"{arr[0]}_{arr[1]}"
        pair0 = arr[2]
        pair1 = arr[3]

        # 检查是否已经处理过
        if already_done_name_file and os.path.isfile(already_done_name_file):
            with open(already_done_name_file, 'r') as done_file:
                already_done_lines = done_file.readlines()
                if any(f"{scene_name}_{pair0}_{pair1}" in line for line in already_done_lines):
                    print(f"{scene_name}_{pair0}_{pair1} already done")
                    continue

        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

        # 执行 Python 脚本
        command = [
            "python", "test_a2pm.py",
            "+experiment=a2pm_mesa_egam_dkm_scannet",
            "test_area_acc=False",
            "test_pm_acc=False",
            "verbose=0",
            f"name={project_name}",
            f"dataset_name={dataset}",
            f"dataset.scene_name={scene_name}",
            f"dataset.image_name0={pair0}",
            f"dataset.image_name1={pair1}"
        ]
        
        # 打印并执行命令
        print("Running command:", " ".join(command))
        subprocess.run(command)
```

**3.**
运行测试程序  
run the benchmark test program

```bash
python mesa-f-dkm-sn.py
```

**4.**
统计metric

```bash
cd ../metric
python eval_ratios.py
```
- NOTE: eval_ratios.py#L21~L26 需要修改到对应路径和文件夹名字.

