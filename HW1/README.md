# NYCU Computer Vision 2025 Spring HW1

StudentID: 313553041  
Name: 盧育霆


## Introduction

My idea is to modify the ResNeXt model by adding CBAM to each layer for channel attention. Leveraging the properties of ResNeXt, the high-dimensional convolutional layers are grouped into multiple identical convolutional layers for convolution operations. Utilizes the channel attention and spatial attention in CBAM to emphasize important channels, suppress unimportant channels, highlight significant spatial regions, and ignore irrelevant background information. In the loss function, in addition to using CrossEntropy, Contrastive Loss is also added.

![Image](https://github.com/user-attachments/assets/b7362dda-e691-4405-aac9-ced973c29f9c)



## How to install
... How to install dependencies


## Performance snapshot
A snapshot of the leaderboard


![Image](https://github.com/user-attachments/assets/dce88b29-aeec-482e-8c92-a0b5d36ff761)
