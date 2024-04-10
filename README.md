# SSESN

The official PyTorch implementation of our **JSTARS 2022** paper:

[**Spatially and Semantically Enhanced Siamese Network for Semantic Change Detection in High Resolution Remote Sensing Images**](https://ieeexplore.ieee.org/document/9736642)



## Release

**Trained Models** (including both bcd and scd models)



## Getting Started

-   ### Environment

    Our experiments are conducted with *python3.6, pytorch1.0.0*, and *CUDA 10.0*.

    Install the requirements using ```pip install -r requirements.txt```

-   ### Data Preparation

    ```
    -dataset
    	|-SECOND
    		|-train
    			|-im1
    			|-im2
    			|-label1			 	(label1_gray to rgb for visualization)
    			|-label2
    			|-label1_gray		(label with 0, 1, 2, ..., 6)
    			|-label2_gray
    			|-mask0_1				(binary mask)
    		|-test
    			|...					 	(same with train)
    	|-CDD
    		|-subset
    			|-train
    				|-A
    				|-B
    				|-OUT
    			|-test
    				|...
    			|-val
    				|...
    ```

    

-   ### Training

    ```shell
    python train_xx.py --checkpointdir ... --datadir ...
    ```

    

-   ### Testing & Visualization

    ```shell
    python visualization.py
    ```

    

-   ### Evaluation

    ```shell
    python SCDD_eval.py
    ```



## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
article{zhao2022spatially,
  title={Spatially and semantically enhanced siamese network for semantic change detection in high-resolution remote sensing images},
  author={Zhao, Manqi and Zhao, Zifei and Gong, Shuai and Liu, Yunfei and Yang, Jian and Xiong, Xiong and Li, Shengyang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={15},
  pages={2563--2573},
  year={2022},
  publisher={IEEE}
}
```

## Contact

If you have any questions or concerns, feel free to open issues or contact me through email [zhaomanqi19@csu.ac.cn].