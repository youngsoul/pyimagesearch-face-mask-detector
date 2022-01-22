python -m mo --reverse_input_channels --batch 1 --mean_values \[127.5, 127.5, 127.5\] --scale_values \[127.5, 127.5, 127.5\] --saved_model_dir ./mask_detector --output_dir openvino_model
rm /Users/patrickryan/.cache/blobconverter/saved_model_openvino_2021.4_6shave.blob
blobconverter --openvino-xml ./openvino_model/saved_model.xml --openvino-bin ./openvino_model/saved_model.bin --shaves 6 --output-dir ./openvino_model --no-cache
