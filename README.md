# Extreme View Synthesis

#### [Paper](https://arxiv.org/abs/1812.04777) | [Extended Presentation at GTC 2019](https://developer.nvidia.com/gtc/2019/video/S9576) (requires free registration) | [Latex citation](#citation)

Code for the paper:  
**Extreme View Synthesis**  
[Inchang Choi](http://www.inchangchoi.info/), [Orazio Gallo](https://oraziogallo.github.io/), [Alejandro Troccoli](https://research.nvidia.com/person/alejandro-troccoli), [Min H. Kim](http://vclab.kaist.ac.kr/minhkim/) and [Jan Kautz](http://jankautz.com/), IEEE International Conference on Computer Vision, 2019 (Oral).


## License

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.

Licensed under the [NVIDIA Source Code License](LICENSE.md)

## Pre-requisites

For convenience, we provide a Dockerfile to build a container image to run the code. The image will contain the Python dependencies and a build of COLMAP.

Your system will need:

1. Docker (>= 19.03)

2. [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker/wiki)

3. NVIDIA GPU driver 418 or later.

Build the container image:

```
docker build -t xtreme-view .
```

## Download the models

You can download the models from the NVIDIA GPU CLOUD registry using:


```
./download_model.sh
```



## Running the code

Place your sequence of images in a directory tree with root ```data```, followed by a directory per sequence, e.g., ```data/0000```, and place all images in the sequence into the ```data/0000/images``` sub-directory.

Launch the container using the provided script:

```
./launch_container.sh
```

Run COLMAP on a sequence of images to get the camera parameters:

```
./run_colmap.sh /data/0000
```

Run the extreme view code generation:

```
python run_xtreme_view.py /data/0000 --input_views=6,8
```

This will run the extreme view synthesis code using images 6 and 8 of the sequence /data/0000. You can modify the code to use different virtual cameras.

You can run COLMAP and the extreme view synthesis on all the sample sequences:

```
./run_colmap_all.sh
./run_xtreme_view_all.sh
```

The results are stored in the sequence directory under ```xtreme-view```. For example, for ```data/0000``` you will find the results in the directory ```data/0000/xtreme-view```. The initial view synthesis is located under ```output``` and the the refined one under ```refinement```.

## <a name="citation"></a> Citation
If you find this code useful in your research or fun project, please consider citing the paper:
```
@inproceedings{extremeview,  
  title={Extreme View Synthesis},  
  author={Choi, Inchang and Gallo, Orazio and Troccoli, Alejandro and Kim, Min H and Kautz, Jan},  
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},  
  pages={7781--7790},  
  year={2019}  
}
```

## Open Source licenses

DeepMVS is Copyright (c) 2018, Po-Han Huang, distributed under the [BSD 2-clause license](https://opensource.org/licenses/BSD-2-Clause)
