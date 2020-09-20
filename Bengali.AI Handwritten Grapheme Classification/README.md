## Bengali.AI Handwritten Grapheme Classification
  A Bengali character has a grapheme root, a vowel and a consonant. The overall goal of this competition is to recognize the root, vowel, and consonant of a given Bengali character image. Information and dataset could be found in https://www.kaggle.com/c/bengaliai-cv19 <br />
<div align=center><img src="https://github.com/jiangdada1221/kaggleCompetition/blob/master/Bengali.AI%20Handwritten%20Grapheme%20Classification/information/Xnip2020-04-13_13-42-56.jpg?raw=true" width = "300" height = "300"/></div> <br />

### Dataset
There are around 20,000 images of Bengali character with labels for training. <br /> There is about the same number of images of Bengali character for testing and they are held by the owner of this competition.

### Method
Due to the similarity of the three tasks - predictions of root, vowel, consonant , I built a multi-task-learning(MTL) model. <br />
The model possessed the hard parameter sharing archetecture (shown in the picture below) <br />
<div align=center><img src="https://example-batch.s3-us-west-1.amazonaws.com/Xnip2020-09-19_22-45-14.jpg" width = "300" height = "300"/></div> <br />
The input image is first processed by a Convolutional Neural Network(CNN) model. Then by the Global Average Pooling, I obtained the feature vector of the given image. At last, the feature vector was transformed into three branches, which were used for the predictions of root, vowel, and consonant respectively. <br />
The base bone CNN model was the EfficientNet https://www.kaggle.com/c/bengaliai-cv19. <br />
Besides, I applied the newly proposed method of data augmentation - GridMask. https://arxiv.org/abs/2001.04086. It's a good data augmentation strategy for image classification. <br />
With some hyperparameters tuning based on the evaluation of the validation dataset (20% of the total training data), I trained a final MTL model, which had a good performance.
https://example-batch.s3-us-west-1.amazonaws.com/Xnip2020-09-19_22-45-14.jpg
### Result- rank 112/2059(top 6%) 
This competition was done all by myself. <br />
At last, I ranked 112/2059 (top 6%)


