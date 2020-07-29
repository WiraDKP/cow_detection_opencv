# Cow Detection using OpenCV
This project is created as celebration of Eid al-Adha 2020. The object detection is performed with Pretrained SSD on COCO dataset. The SSD uses mobilenet v2 as backbone. The video is royalty free and you can found it [here](https://www.videvo.net/video/interested-cows/2867/)

![cow detection](assets/cow_detection.png)
## Step 1: Clone this repository
```
git clone https://github.com/WiraDKP/cow_detection_opencv.git
```

## Step 2: Setup conda environment
Please download miniconda if you haven't. Then use this command to install the environment
```
cd /path/to/cow_detection_opencv
conda env create -f environment.yml
```

## Step 3: Upload your video and setup config
You may use your own video. Upload it to the `data` folder. Then change `FILE_PATH` in `config.py`.<br>
You may also change other configuration as you need.
```
FILE_PATH = "data/cows.mp4"
MODEL_PATH = "model/mobilenet_v2_ssd_coco_frozen_graph.pb"
CONFIG_PATH = "model/mobilenet_v2_ssd_coco_config.pbtxt"
LABEL_PATH = "model/coco_class_labels.txt"

WINDOW_NAME = "detection"

MIN_CONF = 0.4
MAX_IOU = 0.5
```

## Step 4: Activate environment and run the script
```
conda activate cow_detection
python main.py
```