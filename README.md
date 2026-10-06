# Prob-PFNet

The code of paper "Learning the Distribution Evolution of Latent Atmospheric Features for Probabilistic Precipitation Forecasting".

To get the predicted results:

(1) Create a new environment by:

conda create -n py310torch251 gdal python=3.10

pip install torch==2.5.1

pip install tqdm pandas matplotlib opencv-python lpips scikit-image numba zarr xarray

(2) Download the pretrained weight and datasets from: [https://www.kaggle.com/datasets/shuangliangli123/precip_era5_nexrad_mrms-prob](https://www.kaggle.com/datasets/shuangliangli123/precip-era5-nexrad-mrms-prob)

(3) Modify the paths of the dataset and each necessary file.

(4) Run the script: torchrun --nproc_per_node=1 --master_port=55568 main_latent_prob.py --epoch 500 --ex_name '260601_usa-precip-prob-baseline' --batch_size 7 --val_batch_size 1 --test 1 --clip_grad 0 --eval_iter 1000 --save_iter 10 --test_epoch 100

Then the predicted results will be generated in the qualitative_maps directory.

Should you have any questions, please feel free to contact me at whu_lsl@whu.edu.cn.

