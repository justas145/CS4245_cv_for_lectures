# Action Recognition using YOLO and Feature Extraction

This repository contains an implementation for action recognition using the YOLO object detection algorithm, and feature extraction using either OSNet or face recognition.

## Dependencies
- scikit-learn
- PIL
- ultralytics
- numpy
- opencv-python
- argparse
- time

## How to Use
1. Clone the repository
2. Install the required dependencies
3. Run the main script, `main.py`, with your desired arguments.

### Example Usage:
```shell
python main.py -method 'osnet' -detection_confidence 0.8 -num_saved_images 30 -verification_time 2 -threshold_coefficient 0.8 -input_video 'test_videos/test_video3.mp4' 

Arguments
-method: method to use for feature extraction (default: 'osnet')
-detection_confidence: detection confidence threshold (default: 0.75)
-num_saved_images: number of saved images for the target person (default: 30)
-verification_time: time interval between verifications (default: 3)
-threshold_coefficient: coefficient for similarity threshold (default: 0.9)
-save_frames: save frames with the title of similarity score and yes/no for verification (default: False)
-input_video: input video file

### Results_analysis.ipynb
This notebook contains the results of the experiments conducted on the dataset. It also contains the code for the plots and tables in the report.

### Blog Post
See the blog post here: https://medium.com/@justasand1/project-title-bf3dff25a791
