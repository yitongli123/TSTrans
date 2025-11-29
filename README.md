# TSTrans
The official implementation for the paper [_TSTrans: Temporal-Sequence-Driven Transformer for Single Object Tracking in Satellite Videos_].


## Install the environment
Use the Anaconda (CUDA 10.2)
```
conda create -n tstrans python=3.8
conda activate tstrans
cd TSTrans
bash install.sh
```


## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- 1_000000
            |-- 1_000000.rect
            |-- 1_000000.state
            |-- 000001.tiff
            |-- 000002.tiff
            |-- ...
        -- 1_000001
            |-- 1_000001.rect
            |-- 000001.tiff
            |-- 000002.tiff
            |-- ...
        -- ...
        -- 6_000047
        (Note: the above sequences are from SV248S)
        -- car_01
            |-- groundtruth.txt
            |-- img
                |-- 0001.jpg
                |-- 0002.jpg
                |-- ...
        -- ...
        -- car_65
        -- plane_01
        -- ...
        -- plane_09
        -- ship_01
        -- ...
        -- ship_05
        -- train_01
        -- ...
        -- train_26
        (Note: the above sequences are from SatSOT)
        -- aero_039_1
            |-- aero_039_1.txt
            |-- 000001.jpg
            |-- 000002.jpg
            |-- ...
        -- ...
        -- aero_044_2
        -- boat_045_1
        -- ...
        -- boat_047_1
        -- rail_046_1
        -- vehicle_001_1
        -- ...
        -- vehicle_038_1
        (Note: the above sequences are from VISO-SOT)
   ```
*Note:* Each directory (e.g. 1_000001, car_01, vehicle_001_1) represents a sequence. The directory names from SV248S and VISO are changed for the data form of parallel sequences. Specifically, in sequence name '1_000001', the '1' before underline denotes its video id and the '000001' after underline denotes its sequence id from video '01'. In sequence name 'vehicle_001_1', the '001' in the middle denotes its video id and the '1' at the end denotes its sequence id from video '001'. Besides, 'aero, boat, rail, vehicle' represent the categories 'plane, ship, train, car' respectively, for SOT part of VISO.

Put the dataset splits in ./lib/train/data_specs. It should look like this:
    ```
    -- data_specs
        -- satsot
            |-- train_split.txt
            |-- val_split.txt
        -- sv248s
            |-- train_split.txt
            |-- val_split.txt
        -- viso
            |-- train_split.txt
            |-- val_split.txt
    ```


## Training
1. Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/lib/models/tstrans/pretrained_models`.

2. Modify paths in the file `$PROJECT_ROOT$/lib/train/admin/local.py`.

3. Commands:
```
cd TSTrans
(multiple gpus)
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script tstrans --config vitb_256_mae_ce_ep600 --save_dir ./output --mode multiple --nproc_per_node 2
or
(single gpu)
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script tstrans --config vitb_256_mae_ce_ep600 --save_dir ./output --mode single
```
(1) Replace `CUDA_VISIBLE_DEVICES` with your available cuda indices. 
(2) Modify `--save_dir` to be the directory where you plan to save trained networks and tracking results.
(3) Modify the options related to dataset in configuration file `$PROJECT_ROOT$/experiments/tstrans/vitb_256_mae_ce_ep600.yaml`.


## Testing
1. Download the model weights from [Baidu Disk](https://pan.baidu.com/s/1rcVKFFg7m8oypbtb-Iq_-g)(code: isju).

2. Put the downloaded weights corresponding to the tested dataset on `$PROJECT_ROOT$/output/checkpoints/train/tstrans/vitb_256_mae_ce_ep600/`.
Modify the weight name to be `TSTrans_ep$TEST_EPOCH$.pth.tar`($TEST_EPOCH$ in the form of '%04d') according to the setted `TEST.EPOCH` in cofiguration file `vitb_256_mae_ce_ep600.yaml`.

3. Modify paths in the file `$PROJECT_ROOT$/lib/test/evaluation/local.py`.

4. Commands:
```
(parallel mode)
CUDA_VISIBLE_DEVICES=0,1 python tracking/test.py --tracker_name tstrans --tracker_param vitb_256_mae_ce_ep600 --dataset_name satsot_test --threads 2 --num_gpus 2
or
(sequential mode)
CUDA_VISIBLE_DEVICES=1 python tracking/test.py --tracker_name tstrans --tracker_param vitb_256_mae_ce_ep600 --dataset_name satsot_test --threads 0
```
(1) Replace `CUDA_VISIBLE_DEVICES` with your available cuda indices.
(2) Modify `--num_gpus` to be the number of used gpus and the corresponding `--threads`.
(3) Select the `--dataset_name` in ['sv248s_test', 'satsot_test', 'viso_test'] for the tested dataset.


## Evaluation
```
python tracking/analysis_results.py --tracking_results './output/test/tracking_results/tstrans/vitb_256_mae_ce_ep600' --analyzed_split satsot_test
```
(1) Replace `tracking_results` with the path of your saved tracking results to be evaluated.
(2) Select the `--analyzed_split` in ['sv248s_test', 'satsot_test', 'viso_test'] for the evaluated dataset.


## Calculate FLOPs and Model-Scale
*Note:* The speeds reported in our paper were tested on a single NVIDIA GeForce RTX 3090 GPU.
```
python tracking/profile_model.py --script tstrans --config vitb_256_mae_ce_ep600
```


## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) library.
* Thanks for the [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) library.
* The implementation of the ViT is from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.
