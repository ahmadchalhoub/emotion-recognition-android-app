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

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.*;
import org.opencv.videoio.*;
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
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // check that OpenCV has been properly imported
        boolean succesful_import = OpenCVLoader.initDebug();
        Log.d("OpenCVstatus", "Value: " + succesful_import);
    }

    // detect faces in image using Haar Cascade
    public void DetectFace(View v) throws IOException {
        // read input image into Bitmap, convert to OpenCV's Mat, and convert to grayscale
        Bitmap inputImage = BitmapFactory.decodeResource(getResources(), R.drawable.harry_potter);
        Mat source = new Mat(inputImage.getWidth(), inputImage.getHeight(), CvType.CV_8UC4);
        Mat img = new Mat(inputImage.getWidth(), inputImage.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(inputImage, source);
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

        int faces = 0;
        // draw rectangles on original color image
        for (Rect rect : detections.toArray()) {
            Imgproc.rectangle(source, new Point(rect.x, rect.y), new Point(rect.x + rect.width,
                    rect.y + rect.height), new Scalar(255, 0, 0, 0), 30);
            faces = faces + 1;
        }

        // convert final Mat result with rectangles to Bitmap and display result
        ImageView NewImageView = (ImageView)findViewById(R.id.imageView2);
        Bitmap FinalResult;
        FinalResult = Bitmap.createBitmap(source.cols(), source.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(source, FinalResult);
        NewImageView.setImageBitmap(FinalResult);

        String numFaces = faces + " faces were detected";
        TextView textView = (TextView)findViewById(R.id.textView);
        textView.setText(numFaces);
    }

    public void LoadGrayImage(View v) {
        // load image that was clicked into Bitmap variable 'scaled_inputImage' (resized to 48x48)
        Bitmap scaled_inputImage = null;
        String image_name = getResources().getResourceName(v.getId());

        if(image_name.equals(getResources().getResourceName(R.id.angry_image))) {
            scaled_inputImage = Bitmap.createScaledBitmap(BitmapFactory.decodeResource
                    (getResources(), R.drawable.angry_image), 48, 48, true);
        } else if(image_name.equals(getResources().getResourceName(R.id.happy_image))) {
            scaled_inputImage = Bitmap.createScaledBitmap(BitmapFactory.decodeResource
                    (getResources(), R.drawable.happy_image), 48, 48, true);
        }

        ImageView NewImageView = (ImageView)findViewById(R.id.imageView2);
        NewImageView.setImageBitmap(scaled_inputImage);

        // test classification method using grayscale image
        ClassifyEmotion(scaled_inputImage);
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


    // Below 'printBuffer()' function by 'Thomas Klager' from Stackoverflow answer in:
    // https://stackoverflow.com/questions/43273479/java-bytebuffer-slice-not-working-as-per-documentation

    // print byte values of byteBuffer for inspection
    private static  void printBuffer(String prefix,ByteBuffer buff) {
        System.out.println(prefix+buff);
        System.out.println(prefix+Arrays.toString(Arrays.copyOfRange(buff.array(),
                buff.position(), buff.limit())));
    }
}
