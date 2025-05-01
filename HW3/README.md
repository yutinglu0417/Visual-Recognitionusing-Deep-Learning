# NYCU Computer Vision 2025 Spring HW3

StudentID: 313553041  
Name: 盧育霆


## Introduction

In this homework, we were asked to perform instance segmentation on cellular images. The cell dataset contains a total of four classes. First, we need to identify all the different classes of cells in the image. Next, based on the size of the detected cells, we need to draw bounding boxes and calculate the average precision 50. For the model requirements, I used the Mask R-CNN architecture to complete this homework. In this homework, I made some adjustments to the mask predictor and modified several parameters of the region proposal network. These modifications led to a slight improvement in performance.

![Image](https://github.com/user-attachments/assets/b7362dda-e691-4405-aac9-ced973c29f9c)



## How to install

Download the requirements.txt file  
Then do the: pip install -r requirements.txt  
It will auto install numpy, torchtorchvision, matplotlib, opencv-python  


## Performance snapshot
A snapshot of the leaderboard


![Image](https://github.com/user-attachments/assets/dce88b29-aeec-482e-8c92-a0b5d36ff761)
![Image](https://github.com/user-attachments/assets/ee9293af-addb-447e-a64f-b1e97cc32b75)
