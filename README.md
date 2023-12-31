<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#technical-tools">Technical Tools</a></li>
    <li><a href="#data-source">Data source</a></li>
    <li><a href="#the-design">The design</a></li>
    <li><a href="#how-to-use-the-source-code">How to use the source code</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>

### Introduction

This repository hosts the source code for a web application designed for instance segmentation tasks. It detects and segments mitochondria in microscopy images. Detectron2 <a href="https://github.com/facebookresearch/detectron2/blob/main/README.md"> (Wu et al., 2019) </a> was utilized as the primary tool for this task.

### Demo

<p align="center">
  <a href="GIF">
    <img src="media/test_001.png" width="560" alt=""/>
  </a>
</p>

### Technical tools

The application documentation of <a href="https://detectron2.readthedocs.io/en/latest/"> Detectron2 </a> by Facebook research.

-   Pytorch
-   Detectron2
-   YOLOv8
-   numpy
-   pandas
-   opencv-python
-   scikit-image
-   Flask
-   JavaScript (plain)
-   HTML
-   CSS (Bootstrap)
-   Docker

### Data source

This repository hosts the source code for a web application designed for instance segmentation tasks. It detects and segments mitochondria in microscopy images. Two architectures, Detectron2 <a href="https://github.com/facebookresearch/detectron2/blob/main/README.md"> (Wu et al., 2019) </a> and You Only Look Once (i.e., <a href="https://github.com/ultralytics/ultralytics"> YOLOv8 </a>) were utilized in this task for better model comparison.

### The design

I tried three approaches for this task:

-   Using YOLOv8 architecture, <a href="https://github.com/LeoUtas/Mitochondria_segmentation_flask_YOLOv8.git">(source code)</a>;
-   Using Detectron2 architecture, <a href="https://github.com/LeoUtas/Mitochondria_segmentation_flask_detectron2.git">(source code)</a>;
-   Using a combination of the YOLOv8 and Detectron2 architectures, <a href="https://github.com/LeoUtas/Mitochondria_segmentation_flask_2models.git">(source code)</a> (This repository).

In a comparative experiment with this dataset, I evaluated both Detectron2 and YOLOv8. The results indicated that, for this particular task, Detectron2 demonstrated superior performance over YOLOv8 (see <a href="https://mito-app-651cbfda9bde.herokuapp.com/"> comparison images)</a>. However, in some cases, YOLOv8 performed better on the task of object detection.

### How to use the source code

-   Fork/clone this repository (https://github.com/LeoUtas/Mitochondria_segmentation_flask_2models.git)
-   Get the docker container ready

    -   Run docker build (name the app whatever you want on your local machine, and please note that it might take a while for installing all the required dependencies to your local docker image)

    ```cmd
    docker build -t <name of the app> .
    ```

    -   Run the Docker Container (once the docker image is built, you will run a docker container, map it to the port 5000)

    ```cmd
    docker run -p 5000:5000 -v "$(PWD):/app" --name <name of the container> <name of the docker image>
    ```

<br>

I'm thrilled to share it with you all! Dive in and enjoy exploring its features. A big thank you for your interest and for journeying this far with me. Wishing you a fantastic day ahead!

Best,
Hoang Ng

### Reference

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). Detectron2. Retrieved from https://github.com/facebookresearch/detectron2.

Redmon, J. et al., 2016. You Only Look Once: Unified, Real-Time Object Detection. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 779–788.
