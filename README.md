# Prob-PFNet

The code of paper "Learning the Distribution Evolution of Latent Atmospheric Features for Probabilistic Precipitation Forecasting".

To get the predicted results:

(1) Create a new environment by:

conda create -n py310torch251 python=3.10 (3.12 is also OK)

pip install torch==2.5.1 (2.4.0 is also OK)

pip install tqdm==4.67.3 pandas==2.3.3 matplotlib==3.10.8 scikit-image==0.25.2 zarr==2.18.3 xarray==2025.6.1 cartopy==0.26.0 joblib==1.5.3 numcodecs==0.14.1

(2) Download the pretrained weight and datasets from: [https://www.kaggle.com/datasets/shuangliangli123/precip_era5_nexrad_mrms_prob](https://www.kaggle.com/datasets/shuangliangli123/precip-era5-nexrad-mrms-prob) This link provides only a subset of the data. For additional test data, please download them from the official websites of the respective data sources (see Data availability in the paper).

(3) Modify the paths of the dataset and each necessary file.

(4) Run the script: torchrun --nproc_per_node=1 --master_port=55568 main_latent_prob.py --epoch 500 --ex_name '260601_usa-precip-prob-baseline' --batch_size 7 --val_batch_size 1 --test 1 --clip_grad 0 --eval_iter 1000 --save_iter 10 --test_epoch 500

Then the predicted results will be generated in the results directory.

Should you have any questions, please feel free to contact me at whu_lsl@whu.edu.cn.

