package com.example.emotionrecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

// display Camera preview in app
// CameraX Preview setup was obtained from:
// https://developer.android.com/training/camerax/preview
public class CameraXActivity extends MainActivity {

    List<String> labels = Arrays.asList("Angry", "Disgusted", "Afraid", "Happy",
            "Sad", "Surprised", "Neutral");
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView cameraXText;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camerax_activity);

        OpenCVLoader.initDebug();

        previewView = findViewById(R.id.previewView);
        cameraXText = findViewById(R.id.cameraXText);
        imageView = findViewById(R.id.resultImage);

        // request a ProcessCameraProvider
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        // verify that initialization succeeded when View was created
        cameraProviderFuture.addListener(() -> {
            if (checkCameraPermissions(this))
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                }
        }, ContextCompat.getMainExecutor(this));


    }

    // select a camera and bind the lifecycle and use cases
    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetResolution(new Size(480, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            Bitmap bmp = null;
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                @SuppressLint("UnsafeOptInUsageError") Image image = imageProxy.getImage();

                ByteBuffer firstBuffer = image.getPlanes()[0].getBuffer();
                firstBuffer.rewind();
                byte[] firstBytes = new byte[firstBuffer.remaining()];
                firstBuffer.get(firstBytes);
                System.out.println("First Buffer Length = " + firstBytes.length);
                System.out.println("bytes[]: " + Arrays.toString(firstBytes));

                //Create bitmap with width, height, and 4 bytes color (RGBA)
                Bitmap bmp = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
                ByteBuffer buffer = ByteBuffer.wrap(firstBytes);
                bmp.copyPixelsFromBuffer(buffer);
                Bitmap rotatedBMP = null;
                rotatedBMP = rotateBitmap(bmp);
                try {
                    DetectFace(rotatedBMP);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageProxy.close();
            }
        });

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);
    }

    // detect faces in image using Haar Cascade
    public void DetectFace(Bitmap bmp) throws IOException {
        Mat source = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC4);
        Mat img = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(bmp, source);
        Imgproc.cvtColor(source, img, Imgproc.COLOR_RGB2GRAY);

        // read data from 'haarcascade_frontalface_default.xml' file
        // found in 'res/raw/' directory and write data to an output
        // file, cascadeFile, using InputStream and FileOutputStream
        InputStream input_stream = getResources().openRawResource(
                R.raw.haarcascade_frontalface_default);
        File cascadeDir = getDir("haarcascade_frontalface_default", 0);
        File cascadeFile = new File(
                cascadeDir, "haarcascade_frontalface_default.xml");
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

        if (detections.empty()) {
            cameraXText.setText("No faces were detected in the image. Try again!");
            Bitmap bitmapResult;
            bitmapResult = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(),
                    Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(source, bitmapResult);
            imageView.setImageBitmap(bitmapResult);
        } else {
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

            Imgproc.rectangle(source, new Point(highestX, highestY), new Point(highestX +
                            highestWidth, highestY + highestHeight), new Scalar(0, 255, 0));

            Bitmap bitmapResult;
            bitmapResult = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(),
                    Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(source, bitmapResult);
            Bitmap scaledResult = Bitmap.createScaledBitmap(bitmapResult,
                    48, 48, true);
            ClassifyEmotion(scaledResult);
            imageView.setImageBitmap(bitmapResult);
        }
    }

    // rotate the Bitmap returned by camera Intent
    // to normal (vertical) orientation
    public Bitmap rotateBitmap(Bitmap bmp) {
        Matrix matrix = new Matrix();
        matrix.postRotate(90);
        return Bitmap.createBitmap(bmp, 0, 0,
                bmp.getWidth(), bmp.getHeight(), matrix, true);
    }

    public void ClassifyEmotion (Bitmap detected_image) {
        try {
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

            tflite.run(newTensorImage.getBuffer(), probabilityBuffer.getBuffer());
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

        String finalResult = "The face is '" + labels.get(maxIndex) + "' with " +
                "classification " + "value of " + df.format(maxValue*100) + " %";
        cameraXText.setText(finalResult);
    }


    // onClick function to allow user to return to MainActivity
    public void returnToHome(View v) {
        Intent returnIntent = new Intent(this, MainActivity.class);
        startActivity(returnIntent);
    }

    // this function was obtained from (with slight changes from me):
    // https://stackoverflow.com/questions/67553067/cannot-open-camera-0-without-camera-permission
    public boolean checkCameraPermissions(Context context){
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED)
        {
            // Permission is not granted
            Log.d("checkCameraPermissions", "No Camera Permissions");
            ActivityCompat.requestPermissions((Activity) context,
                    new String[] { Manifest.permission.CAMERA },
                    100);
            return false;
        }
        return true;
    }


}
