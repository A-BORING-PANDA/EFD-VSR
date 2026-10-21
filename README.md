## EFD-VSR
code for "Illuminating the Dark: Event-Flow Guided Deformable Alignment for Low-Light Video Super-Resolution"

### Installation
* Dependencies: [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), [CUDA Toolkit 11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive), [torch 1.10.2+cu111](https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl), and [torchvision 0.11.3+cu111](https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl).

### Test
We have uploaded the pretrained model and the test dataset. The test dataset is a subset of the DSEC dataset.
The pretrain model, visual result, dataset is in [Google](https://drive.google.com/drive/folders/1qKv449eNe09bUvm23Wy4rzymw6LQ7R1l?usp=drive_link)
Please put the folder in
- Pretrained model: `~/experiments/EFDVSR_x4/models/net_g_latest.pth`
- Test dataset: `~/dataset/test`

3. Run the following command:
    * Test on DSEC_new for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh 1 options/test/test_EFDVSR_x4.yml
      ```


###  Data Structure
* Both video and event data are required as input. We package each video and its event data into an [HDF5](https://docs.h5py.org/en/stable/quick.html#quick) file.
* The Low-Light (LR) HDF5 file contains `images` and `voxels`.
* The Ground-Truth (GT) HDF5 file contains `images`.

* Example: The structure of an HDF5 file:
  ```
  ClipName.h5
  ├── images
  │   ├── 000000 # frame, ndarray, [H, W, C]
  │   ├── ...
  ├── voxels
  │   ├── 000000 # event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ```


### License and Acknowledgement
This project is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Our implementation incorporates code adapted from [EvTexture](https://github.com/DachunKai/EvTexture) ; [Ev-DeblurVSR](https://github.com/DachunKai/Ev-DeblurVSR);  [RetinexEVSR](https://github.com/DachunKai/RetinexEVSR). We would also like to sincerely thank the developers of [BasicSR](https://github.com/XPixelGroup/BasicSR), an open-source toolbox for image and video restoration tasks. Additionally, we appreciate the inspiration and code provided by [event_utils](https://github.com/TimoStoff/event_utils); [MotionPriorCMax](https://github.com/tub-rip/MotionPriorCMax). We gratefully acknowledge the use of the DSEC dataset, which is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). We also thank the authors for their contribution to the community. The implementation in this repository is adapted from RetinexEVSR with minor modifications.
