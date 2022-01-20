# PyImageSearch Face Mask Detection

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

Working through the blog post on how to use Fine-Tuning Transfer Learning to detect if someone is wear a mask in an image or in a video feed.


## Additional Resources

https://towardsdatascience.com/real-time-face-mask-detector-with-tensorflow-keras-and-opencv-38b552660b64

## Setup

```shell
pip install opencv-python
pip install tensorflow
pip install imutils
pip install scikit-learn
pip install matplotlib

pip install openvino-dev
```

## Convert Tensorflow Model to OpenVino IR
https://docs.luxonis.com/en/latest/pages/model_conversion/

This will allow the model to run on OAK Device

```shell
python -m mo --saved_model_dir ./mask_detector --output_dir openvino_model --framework tf --input_shape=\[1,224,224,3\]
```

After this runs, you should have a directory with an .xml, .bin, .mapping file.  The .xml and .bin can be used with the [Luxonis Myriad Online Compiler](http://blobconverter.luxonis.com) to create a .blob file

or

```shell
python -m blobconverter --openvino-xml ./openvino_model/saved_model.xml --openvino-bin ./openvino_model/saved_model.bin --shaves 6
```

## Train Mask Model

`pr_train_model.sh`

## Image

`pr_detect_image.sh`

## Video

run `pr_detect_mask_video.py`

