# emotion-recognition-android-app

This is a native Android Application, built in Android Studio, that uses OpenCV's Haar Cascade and a Convolutional Neural Network to perform facial emotion recognition. 
The seven different emotions that the CNN is trained for are the following:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

This is a custom trained model (steps for training shown here):
https://github.com/ahmadchalhoub/emotion-recognition-haar-cascade

The trained model has the following metrics: 96.67% Training Accuracy | 53.64% Testing Accuracy

Once the basic infrastructure for the application is done and usability is good, the CNN will be retrained 
for a higher testing accuracy (better generalization) so that the user has a better experience. 
