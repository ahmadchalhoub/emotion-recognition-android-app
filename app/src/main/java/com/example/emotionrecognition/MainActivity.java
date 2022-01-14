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
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.*;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST = 100;
    private Button captureButton;
    private ImageView imageView;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // check that OpenCV has been properly imported
        boolean succesful_import = OpenCVLoader.initDebug();
        Log.d("OpenCVstatus", "Value: " + succesful_import);

        captureButton = findViewById(R.id.captureFrame);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent open_camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(open_camera, 100);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST) {
            if (data.equals(null)) {
                System.out.println("Your data is = null");
            } else if (data != null) {
                System.out.println("Your data is not = null");
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                Bitmap rotatedPhoto = null;
                rotatedPhoto = rotateBitmap(photo);
                try {
                    DetectFace(rotatedPhoto);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // rotate the Bitmap returned by camera Intent
    // to normal (vertical) orientation
    public Bitmap rotateBitmap(Bitmap bmp) {
        Matrix matrix = new Matrix();
        matrix.postRotate(90);
        Bitmap rotatedPhoto = null;
        rotatedPhoto = Bitmap.createBitmap(bmp, 0, 0,
                bmp.getWidth(), bmp.getHeight(), matrix, true);
        return rotatedPhoto;
    }

    // detect faces in image using Haar Cascade
    public void DetectFace(Bitmap bmp) throws IOException {

            imageView.setImageBitmap(bmp);
            Mat source = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC4);
            Mat img = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(bmp, source);
            Imgproc.cvtColor(source, img, Imgproc.COLOR_RGB2GRAY);

            // read data from 'haarcascade_frontalface_default.xml' file found in 'res/raw/' directory and
            // write data to an output file, cascadeFile, using InputStream and FileOutputStream
            InputStream input_stream = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = getDir("haarcascade_frontalface_default", 0);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream output_stream = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesTransferred;
            while ((bytesTransferred = input_stream.read(buffer)) != -1) {
                output_stream.write(buffer, 0, bytesTransferred);
            }
            input_stream.close();
            output_stream.close();

            // declare CascadeClassifier object using absolute
            // path of newly created XML file
            CascadeClassifier faceDetector = new CascadeClassifier(
                    cascadeFile.getAbsolutePath());
            MatOfRect detections = new MatOfRect();

            // perform face detections
            faceDetector.detectMultiScale(img, detections);

            // extract values for biggest detected face
            double highestArea = Integer.MIN_VALUE;
            int highestX = Integer.MIN_VALUE;
            int highestY = Integer.MIN_VALUE;
            int highestWidth = Integer.MIN_VALUE;
            int highestHeight = Integer.MIN_VALUE;
            for (Rect rect : detections.toArray()) {
                if (rect.area() > highestArea) {
                    highestArea = rect.area();
                    highestX = rect.x;
                    highestY = rect.y;
                    highestWidth = rect.width;
                    highestHeight = rect.height;
                }
            }

            Rect rectCrop = new Rect(highestX, highestY, highestWidth, highestHeight);
            Mat croppedImage = new Mat(img, rectCrop);

            Bitmap croppedResult;
            croppedResult = Bitmap.createBitmap(highestWidth, highestHeight,
                    Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedImage, croppedResult);
            imageView.setImageBitmap(croppedResult);

            Bitmap scaledResult = Bitmap.createScaledBitmap(croppedResult,
                    48, 48, true);
            ClassifyEmotion(scaledResult);
    }

    public void ClassifyEmotion (Bitmap detected_image) {
        try {

            List<String> labels = Arrays.asList("Angry", "Disgust", "Fear", "Happy",
                    "Sad", "Surprise", "Neutral");

            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(
                    this.getApplicationContext(), "emotion_cnn.tflite");

            Interpreter tflite = new Interpreter(tfliteModel);

            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new TransformToGrayscaleOp())
                    .build();

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(detected_image);
            TensorImage newTensorImage = imageProcessor.process(tensorImage);
            TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(
                    new int[]{1, 7}, DataType.FLOAT32);

            if(null != tflite){
                tflite.run(newTensorImage.getBuffer(), probabilityBuffer.getBuffer());
            }

            getClassification(probabilityBuffer.getFloatArray(), labels);

        } catch (IOException e) {
            System.out.println("Image classification failed!");
        }
    }

    // print the biggest classification probability and its corresponding index
    private void getClassification(float[] floatArray, List<String> labels){

        DecimalFormat df = new DecimalFormat("0.00");
        float maxValue = Integer.MIN_VALUE;
        int maxIndex = 0;

        int index = 0;
        while( index < floatArray.length ) {
            if( maxValue < floatArray[index] ) {
                maxValue = floatArray[index];
                maxIndex = index;
            }
            index++;
        }

        String finalResult = "The face is '" + labels.get(maxIndex) +
                "' with classification " + "value of " + df.format(maxValue*100) + " %";

        TextView textView = (TextView)findViewById(R.id.textView);
        textView.setText(finalResult);
    }
}
