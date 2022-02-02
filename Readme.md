# PyImageSearch Face Mask Detection

## Attribution
Adrian Rosebrock, 
COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning, 
PyImageSearch, 
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

## Overview
Working through the blog post on how to use Fine-Tuning Transfer Learning to detect if someone is wear a mask in an image or in a video feed.


## Additional Resources

https://towardsdatascience.com/real-time-face-mask-detector-with-tensorflow-keras-and-opencv-38b552660b64

## Setup

Note:  You only need the openvino-dev you plan to try to take this to a system that is using OpenVino such as the OAK devices from OpenCV AI Kit, Luxonis.

```shell
pip install opencv-python
pip install tensorflow
pip install imutils
pip install scikit-learn
pip install matplotlib

pip install openvino-dev
pip install blobconverter
```

## Convert Tensorflow Model to OpenVino IR
https://docs.luxonis.com/en/latest/pages/model_conversion/

This will allow the model to run on OAK Device

See the script: `create_openvino.sh`

```shell
python -m mo --reverse_input_channels --batch 1 --mean_values \[127.5, 127.5, 127.5\] --scale_values \[127.5, 127.5, 127.5\] --saved_model_dir ./mask_detector --output_dir openvino_model
```

After this runs, you should have a directory with an .xml, .bin, .mapping file.  

The .xml and .bin can be used with the [Luxonis Myriad Online Compiler](http://blobconverter.luxonis.com) to create a .blob file

or

```shell
blobconverter --openvino-xml ./openvino_model/saved_model.xml --openvino-bin ./openvino_model/saved_model.bin --shaves 6 --output-dir ./openvino_model --no-cache
```

## Train Mask Model

`pr_train_model.sh`

## Detect mask in an image

`pr_detect_image.sh`

## Detect mask in webcam video

run `pr_detect_mask_video.py`

