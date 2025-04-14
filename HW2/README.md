# NYCU Computer Vision 2025 Spring HW2

StudentID: 313553041  
Name: 盧育霆


## Introduction

In this homework, we were asked to perform house number recognition. First, we need to locate the numbers in the image and find appropriate bounding boxes to enclose them. Next, we need to recognize the numbers in the image. For the model requirements, we are only allowed to use the Faster R-CNN architecture. We can only modify the backbone network, region proposal network, and head. In this homework, I made adjustments to the region proposal network and do some data augmentation. Expert to improve the performance.

![Image](https://github.com/user-attachments/assets/f3e3836d-16f4-40bf-a2b9-f6bd0161364d)


## How to install

Download the requirements.txt file  
Then do the: pip install -r requirements.txt  
It will auto install numpy, torchtorchvision, matplotlib, opencv-python  


## Performance snapshot
A snapshot of the leaderboard


![Image](https://github.com/user-attachments/assets/dce88b29-aeec-482e-8c92-a0b5d36ff761)
![Image](https://github.com/user-attachments/assets/ee9293af-addb-447e-a64f-b1e97cc32b75)
