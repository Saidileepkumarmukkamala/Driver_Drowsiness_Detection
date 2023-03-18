# Driver_Drowsiness_Detection
## Introduction:
Driver drowsiness is a significant concern and can lead to accidents on the road. In recent years, there has been a lot of interest in developing driver drowsiness detection systems. One of the most effective ways to detect driver drowsiness is by using computer vision techniques. In this project, we will use Convolutional Neural Networks (CNN) to detect driver drowsiness and play an alarm sound to alert the driver and We used streamlit to create a website and deployed it to make it access to the public.

### Dataset:
To train our CNN model, we will need a dataset of images of drivers with both drowsy and awake states. One of the popular datasets used for this task is the 'Drowsiness Detection Dataset' that contains 14,941 images of drivers in different drowsy states. The dataset is labeled into two classes: drowsy and awake.

### Preprocessing:
We will start by preprocessing the images before feeding them to the CNN model. We will first resize the images to a standard size (e.g., 224x224 pixels) and then normalize the pixel values between 0 and 1.

### Model Architecture:
For our CNN model, we will use a pre-trained architecture such as VGG16 or ResNet50 as they have been shown to perform well on image classification tasks. We will remove the top layers of the pre-trained model and add new layers that are specific to our task. The final layer will be a dense layer with a single output unit and a sigmoid activation function to predict the probability of the driver being drowsy.

### Training:
We will split the dataset into training and validation sets and train the model using the Adam optimizer and binary cross-entropy loss function. We will also use data augmentation techniques such as rotation, zoom, and horizontal flip to increase the size of the dataset and reduce overfitting.

### Testing:
To test the performance of our model, we will use a separate test set and calculate the accuracy, precision, recall, and F1-score. If the model achieves satisfactory performance, we can move on to the final step.

### Alarm Sound:
Once the model detects driver drowsiness, it will trigger an alarm sound to alert the driver. We can use a pre-recorded audio file or generate a sound programmatically. We can use libraries such as pyaudio or playsound to play the audio file.

### Conclusion:
In this project, we have developed a driver drowsiness detection app using streamlit, CNN and deployed using streamlit cloud, It can trigger an alarm sound when the driver is drowsy. This system can potentially reduce the number of accidents caused by driver drowsiness on the road. Further improvements can be made by incorporating other sensors such as steering angle, speed, and eye-tracking to enhance the accuracy of the system.
