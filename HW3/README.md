# NYCU Computer Vision 2025 Spring HW3

StudentID: 313553041  
Name: 盧育霆


## Introduction

In this homework, we were asked to perform instance segmentation on cellular images. The cell dataset contains a total of four classes. First, we need to identify all the different classes of cells in the image. Next, based on the size of the detected cells, we need to draw bounding boxes and calculate the average precision 50. For the model requirements, I used the Mask R-CNN architecture to complete this homework. In this homework, I made some adjustments to the mask predictor and modified several parameters of the region proposal network. These modifications led to a slight improvement in performance.

![Image](https://github.com/user-attachments/assets/28a4c4bc-b6c4-4350-ad61-b64150010532)

![Image](https://github.com/user-attachments/assets/8095b271-c03d-43f1-8b63-2835d9db18db)


## How to install

Download the requirements.txt file  
Then do the: `pip install -r requirements.txt`  
It will auto install numpy, torchtorchvision, matplotlib, opencv-python  

Execution: `python main.py`  
Note: you should generate all mask in dataset, shown as Figure 1.  
And the image naming rule is [class_num]_[num_pic].tif.  
![Image](https://github.com/user-attachments/assets/43fff835-4a38-4fa7-aaec-27906bc517d5)
Figure 1. example of dataset form. 

## Performance snapshot
A snapshot of the leaderboard


![Image](https://github.com/user-attachments/assets/f7994e65-f65f-4433-9032-2c4b538b8f31)
![Image](https://github.com/user-attachments/assets/964d9161-8e4b-4dfe-b1c7-21338b391bf5)
