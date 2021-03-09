# Flipkart Attribute Prediction on Fashion dataset
Using a 26-class Resnext classifier initialized with pretrained model (which is trained on ILSVRC dataset - 1.2M natural images with 1000 classes) for classifying categories (26 categories available).

# Methodology
1. Train ResNext-50 classifier on 26 categories. This is an easy problem. We use Weighted Categorical Cross Entropy loss and Adam optimizer with beta1 = 0.5, beta2 = 0.999. The learning rate starts at 2e-3 and reduces by half when loss plateaus.
2. Remove the last layer from the ResNext-50 which is fully trained on category classification, and replace it with the 46 MLP (Multi-Layer Perceptron) heads (since there are 46 attribute keys) each predicting attribute values for a particular attribute key. This is a multi-label classification problem (which is a relatively hard problem).
3. The MLP used here has 2048 input neurons, 256 hidden neurons and the output neurons are as many classes as required.
4. The entire 46-headed ResNext-50 is trained with Binary Cross Entropy Loss (we wanted to do weighted loss here as well, but couldn't do that because of time constraints) for multi-label classification.

# Implementation details

1. Run "train.py" with custom options for data directories, number of epochs, batch size, learning rate, etc. This will save a ResNext-50 model for category classification. 
2. The inference can be performed by loading this model from "test.py".
3. Then run "train_att.py" to load the ResNext-50 category classifier, which is already trained on categories, and train the whole 46-headed ResNext-50 model.
4. Finally the inference on attribute value prediction can be performed on this model using "test_att.py"
5. In all predictions, we threshold the output for attribute prediction at 0.8 (this can be changed), since the last layer is a sigmoid.

# Conclusion
1. We divided the entire data into train (~75%) and validation (~25%) keeping the same distribution of categories in both.
2. We trained the ResNext-50 category classifier for 13 epochs and 46-headed ResNext-50 for 8 epochs, finally we fonud that in the validation set we got accuracy of ~64%.
3. We find that the accuracy (IOU) of our 46-headed ResNext-50 stalled at a mere 64%. And it predicts a lot of unnecessary attributes which makes it less accurate.
