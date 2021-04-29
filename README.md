# flower-classifier-CNN

Queries user about whether to download creative-commons licensed flower photos 
from Google or to specify the path of one existing locally. Furthermore,
queries to split data into training and validation subsets or train on the 
whole dataset. Finally, trains a convolutional neural network constructed using
the VGG16 CNN as the base model (with  imagenet's pre-trained weights but 
without its fully connected top layer) with these additional layers at the top:
    1. Flatten
    2. Fully connected hidden layer with 256 neurons
    3. Softmax layer
to predict whether a given image (resized to 150 px by 150 px) is of a daisy,
dandelion, rose, sunflower, or tulip.


As can be seen in the log files, the model with all layers left untrainable provided the best accuracy over 10 epochs. This is because images of all classes of flowers, except dandelion, were used as training data in imagenet's trainig of vgg16, therefore, when training to classify this set of flowers, the weights in the filters are already highly optimized to distinguish between them. Increasing the amount of trainable layers actually decreases the accuracy when this short number of epochs is used. This is because the model's weights become updated with each batch and in the process of finding the new filters corresponding to the new minima in accuracy error, the weights become tuned in such a way that lowers their accuracy, however, with a large enough dataset and enough epochs, the accuracy should theoretically improve to beyond the model with all layers left untrainable.
