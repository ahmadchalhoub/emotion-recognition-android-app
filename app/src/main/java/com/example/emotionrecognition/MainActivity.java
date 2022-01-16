// Author: Ahmad Chalhoub (https://github.com/ahmadchalhoub)

// Description: This is a native Android Application, built in Android Studio,
// that uses OpenCV's Haar Cascade and a Convolutional Neural Network to perform
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
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST = 100;
    private TextView chosenCamera;

    // default (initial) rotation angle for Bitmap returned
    // by camera Intent is 90 (back camera). This is considering
    // that the user didn't choose one of the two options
    // provided for camera choice (front or back)
    int rotationAngle = 90;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // check that OpenCV has been properly imported
        boolean succesful_import = OpenCVLoader.initDebug();
        Log.d("OpenCVstatus", "Value: " + succesful_import);

        Button captureButton = findViewById(R.id.captureFrame);
        chosenCamera = findViewById(R.id.chosenCamera);

        captureButton.setOnClickListener(v -> {
            Intent open_camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(open_camera, 100);

        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST) {
            if (data != null) {
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                Bitmap rotatedPhoto = rotateBitmap(photo);
                Intent resultIntent = new Intent(MainActivity.this, ResultActivity.class);
                resultIntent.putExtra("rotatedPhoto", rotatedPhoto);
                startActivity(resultIntent);
            }
        }
    }

    // rotate the Bitmap returned by camera Intent
    // to normal (vertical) orientation
    public Bitmap rotateBitmap(Bitmap bmp) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationAngle);
        return Bitmap.createBitmap(bmp, 0, 0,
                bmp.getWidth(), bmp.getHeight(), matrix, true);
    }

    // determine which camera user wants to use (front/back) and
    // set rotationAngle accordingly
    public void selectCamera(View v) {
        Button b = (Button)v;
        String textButton = b.getText().toString();
        switch (textButton) {
            case "Front Camera":             // front camera selected
                rotationAngle = 270;
                chosenCamera.setText("You will be using the front camera.");
                break;
            case "Back Camera":             // back camera selected
                rotationAngle = 90;
                chosenCamera.setText("You will be using the back camera.");
                break;
        }
    }
}
