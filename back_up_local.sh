
###
 # @Author: Easonyesheng preacher@sjtu.edu.cn
 # @Date: 2025-09-10 17:30:10
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2025-09-10 17:34:57
 # @FilePath: /A2PM-MESA/back_up_local.sh
 # @Description: copy all code to ../A2PM-MESA_local
### 

dst=../A2PM-MESA_local
except="--exclude .git \
        --exclude data \
        --exclude .vscode \
        --exclude __pycache__ \
        --exclude *.pyc \
        --exclude *.png \
        --exclude *.jpg \
        --exclude *.ply \
        --exclude *.npy \
        --exclude *.JPG \
        --exclude *.h5 \
        --exclude *.log \
        --exclude *.bin \
        --exclude *.db \
        --exclude *.pth \
        --exclude *.ckpt \
        --exclude *.zip \
        --exclude *.tar \
        --exclude *.tar.gz \
        "

rsync -av $except ./ $dst
