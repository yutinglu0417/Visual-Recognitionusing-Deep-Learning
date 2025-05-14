# NYCU Computer Vision 2025 Spring HW4

StudentID: 313553041  
Name: 盧育霆


## Introduction

In this homework, we were asked to implement image denoising. There are two types of image noise in the dataset: rain and snow. For the model requirements, we must use an all-in-one model to denoise images with both types of noise. I used PromptIR [1] to complete this assignment. This model utilizes prompts to achieve all-in-one image denoising. In additional experiments, I modified the prompt architecture and the shallow layer structure. In the final results, the Peak Signal-to-Noise Ratio (PSNR) over 30.

![Image](https://github.com/user-attachments/assets/b7362dda-e691-4405-aac9-ced973c29f9c)



## How to install

Download the requirements.txt file  
Then do the: pip install -r requirements.txt  
It will auto install numpy, torchtorchvision, matplotlib, opencv-python  


## Performance snapshot
A snapshot of the leaderboard


![Image](https://github.com/user-attachments/assets/dce88b29-aeec-482e-8c92-a0b5d36ff761)
![Image](https://github.com/user-attachments/assets/ee9293af-addb-447e-a64f-b1e97cc32b75)
