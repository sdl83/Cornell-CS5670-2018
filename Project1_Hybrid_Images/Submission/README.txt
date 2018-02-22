Sarah Le Cam - sdl83

Project 1: Hybrid Images

Low Pass
Image: left.png -- frowning image
Sigma size: 8.0
Kernel Size: 10

High Pass
Image: right.png -- smiling image
Sigma size: 7.2
Kernel Size: 12


Mix-In Ratio: 0.6


In this lab, we created a hybrid image by combining a low pass and high pass picture. In our code implementation, we created the high pass and low pass filters by implementing cross correlation through convolution with a Gaussian kernel. To create the hybrid image, I chose to use a picture of myself smiling and frowning and was careful to move as little as possible between pictures to maximize alignment. I found it difficult to find the right balance so that the low pass image would show at a distance without altering the close up (high pass) image, but played with the parameters until I found an appropriate balance.


