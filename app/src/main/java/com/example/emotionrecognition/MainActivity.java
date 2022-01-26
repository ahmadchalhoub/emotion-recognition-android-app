// Author: Ahmad Chalhoub (https://github.com/ahmadchalhoub)

// Description: This is a native Android Application, built in Android Studio,
// that uses Google's MLKit and a Convolutional Neural Network to perform
// facial emotion recognition. The seven different emotions that the CNN is trained
// for are the following:
//
//              1. Angry
//              2. Disgust
//              3. Fear
//              4. Happy
//              5. Sad
//              6. Surprise
//              7. Neutral

// This is a custom trained model (steps for training shown in below link):
// https://github.com/ahmadchalhoub/emotion-recognition-haar-cascade

// The trained model has the following metrics: 96.67% Training Accuracy | 53.64% Testing Accuracy

// Progress: So far, this application is able to do two things:
//
//              1. Load in one of two 48*48 grayscale images and perform emotion classification
//                 on them (CNN), and outputting a 'class' and a 'probability percentage accuracy'
//              2. Load an RGB image with multiple faces in it, detect all the faces in the image
//                 using OpenCV's Haar Cascade, and outputting an image with bounding boxes on
//                 the detected faces

// The end goal is for the application to take in a live camera feed, detect all faces,
// classify the faces' emotions, and output a live, real-time bounding box on each face
// with its corresponding emotion classification

package com.example.emotionrecognition;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    // open CameraXActivity to perform and view emotion recognition
    public void openCameraX(View v) {
        Intent returnIntent = new Intent(this, CameraXActivity.class);
        returnIntent.putExtra("from", "MainActivity");
        startActivity(returnIntent);
    }
}