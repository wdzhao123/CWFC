# Center-Wise Feature Consistency Learning for Long-Tailed Remote Sensing Object Recognition
  Implementation of "Center-Wise Feature Consistency Learning for Long-Tailed Remote Sensing Object Recognition" in PyTorch

## Preparation
### Datasets
  We conduct experiments on four remote sensing datasets: FGSC-23, DIOR, HRSC2016 and xView.

  - `FGSC`: Contains 4,081 images from 23 categories, 3,256 samples for training and 825 samples for testing.  
  - `DIOR`: Contains 192,465 images from 20 categories, 68,025 samples for training and 124,440 samples for testing.
  - `HRSC`: Contains 2,076 images from 18 categories, 1,172 samples for training and 904 samples for testing.
  - `xView`: Contains 98,832 images from 51 categories, 69,206 samples for training and 29,626 samples for testing.

  You can download the preprocessed datasets from [Baidu Netdisk](https://pan.baidu.com/s/1xQFNwlIa_cIKEQIyoIiHZQ?pwd=bc6e) or [Google Drive](https://drive.google.com/drive/folders/1Wonc7KJhshIT2WLY23k5o86wIASXsK_A?usp=sharing), then extract them to `dataset/`.

## Training
  ```
  cd CWFC/code/
  ```

### Train teacher models
  Train both head and tail teacher models using  
  ```
  python train_t.py --model_save_dir MODEL_SAVE_DIR --data_dir dataset/DATASET_NAME --threshold T
  ```
  where `MODEL_SAVE_DIR` refers to the directory for saving the trained models, `DATASET_NAME` is the name of target dataset, `T` stands for the threshold to divide the head and tail subsets.

### Train student model
  After obtaining the teacher models, train the student model using
  ```
  python train_s.py --head_teacher_path HEAD_TEACHER_PTH --tail_teacher_path TAIL_TEACHER_PTH --model_save_dir MODEL_SAVE_DIR --data_dir dataset/DATASET_NAME --threshold T
  ```
  where `HEAD_TEACHER_PTH` and `TAIL_TEACHER_PTH` are the trained teacher models on both subsets.

## Testing
  ```
  cd CWFC/code/
  ```

  Evaluate the recognition accuracy of the trained model using
  ```
  python test.py --model_path MODEL_PTH --data_dir dataset/DATASET_NAME
  ```
  where `MODEL_PTH` refers to the trained model for evaluation, `DATASET_NAME` is the name of target dataset.

  We also provide our pretrained models on all four datasets for reference. You can download them from [Baidu Netdisk](https://pan.baidu.com/s/1tbdSEUR82GCIlCfw8RYTxw?pwd=i7i1) or [Google Drive](https://drive.google.com/drive/folders/1M1zU6niofurHY4nvCCMFJavrBPSeQ9z6?usp=sharing).

# Cite

  If you find our CFCL useful in your research, please consider citing it: 

  ```
  @article{zhao2024center,
  title={Center-Wise Feature Consistency Learning for Long-Tailed Remote Sensing Object Recognition},
  author={Zhao, Wenda and Zhang, Zhepu and Liu, Jiani and Liu, Yu and He, You and Lu, Huchuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
  }
  ```
