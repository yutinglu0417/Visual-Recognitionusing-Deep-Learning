# NYCU Computer Vision 2025 Spring HW4

StudentID: 313553041  
Name: 盧育霆


## Introduction

In this homework, we were asked to implement image denoising. There are two types of image noise in the dataset: rain and snow. For the model requirements, we must use an all-in-one model to denoise images with both types of noise. I used PromptIR to complete this assignment. This model utilizes prompts to achieve all-in-one image denoising. In additional experiments, I modified the prompt architecture and the shallow layer structure, respectively shown in figure 2, 3. In the final results, the Peak Signal-to-Noise Ratio (PSNR) over 30.

![Image](https://github.com/user-attachments/assets/f553f666-ef7a-41f5-939f-7d599f90aebd)  
Figure 1. architecture of PromptIR

![Image](https://github.com/user-attachments/assets/225cc727-898d-45b3-8e93-246751f2b2bb)   
Figure 2. new prompt generation module

![Image](https://github.com/user-attachments/assets/78602e8f-e5e9-47c3-a76c-b0f8ab329fe2)  
Figure 3. new feature encoder

![Image](https://github.com/user-attachments/assets/03adbe9d-11cc-4c76-a7c8-5fcb5ca72071)  
Figure 4. visualization of result

## How to install

Download the requirements.txt file
Then do the: pip install -r requirements.txt
It will auto install numpy, torchtorchvision, matplotlib, opencv-python

Execution: python main.py


## Performance snapshot
A snapshot of the leaderboard

![Image](https://github.com/user-attachments/assets/ca5a019d-8ef8-4868-97ea-7572cd9c2ace)
![Image](https://github.com/user-attachments/assets/fb106109-0326-490c-a86b-493bc5d17bb8)
