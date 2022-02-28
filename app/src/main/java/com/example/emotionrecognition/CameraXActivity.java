package com.example.emotionrecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;

import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.os.Bundle;
import android.renderscript.ScriptGroup;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceDetector;

import org.opencv.android.OpenCVLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
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

    // list of classes the trained CNN can detect
    List<String> labels = Arrays.asList("Angry", "Disgusted", "Afraid", "Happy",
            "Sad", "Surprised", "Neutral");

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView cameraXText;
    private Boolean frontCamera;
    private ImageView imageView;
    private boolean flipBox;
    private GraphicOverlay mGraphicOverlay;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camerax_activity);

        OpenCVLoader.initDebug();

        previewView = findViewById(R.id.previewView);
        cameraXText = findViewById(R.id.cameraXText);
        imageView = findViewById(R.id.imageView);
        mGraphicOverlay = findViewById(R.id.graphic_overlay);

        Intent intent = getIntent();
        String previousActivity = (String) intent.getExtras().get("from");

        // use back camera if coming from MainActivity.
        // flip camera if coming from this same activity (after pressing
        // the 'FLIP' button)
        if (previousActivity.equals("MainActivity")) {
            frontCamera = true;
        } else {
            Boolean newCameraChosen = (Boolean)intent.getExtras().get("cameraChosen");
            frontCamera = !newCameraChosen;
        }

        // request a ProcessCameraProvider
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        // verify that initialization succeeded when View was created
        cameraProviderFuture.addListener(() -> {
            if (checkCameraPermissions(this))
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider);
                } catch (ExecutionException | InterruptedException ignored) {
                }
        }, ContextCompat.getMainExecutor(this));
    }

    // select a camera and bind the lifecycle and use cases
    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector;

        if (frontCamera) {
            cameraSelector = new CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                    .build();
        } else {
            cameraSelector = new CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                    .build();
        }

        // Configure the face detector
        FaceDetectorOptions realTimeOpts = new FaceDetectorOptions.Builder()
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(new Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), imageProxy -> {
            @SuppressLint("UnsafeOptInUsageError") Image image = imageProxy.getImage();
            assert image != null;

            ByteBuffer firstBuffer = image.getPlanes()[0].getBuffer();
            firstBuffer.rewind();
            byte[] firstBytes = new byte[firstBuffer.remaining()];
            firstBuffer.get(firstBytes);

            // Store image from camera into a Bitmap
            Bitmap bmp = Bitmap.createBitmap(image.getWidth(), image.getHeight(),
                    Bitmap.Config.ARGB_8888);
            ByteBuffer buffer = ByteBuffer.wrap(firstBytes);
            bmp.copyPixelsFromBuffer(buffer);
            InputImage bmpImage = InputImage.fromBitmap(bmp,
                    imageProxy.getImageInfo().getRotationDegrees());

            // initialize detector
            FaceDetector detector = FaceDetection.getClient(realTimeOpts);
            Task<List<Face>> task = detector.process(bmpImage)
                    .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                        @Override
                        public void onSuccess(List<Face> faces) {
                            Bitmap rotatedBMP = rotateBitmap(bmp,
                                    imageProxy.getImageInfo().getRotationDegrees());

                            Bitmap croppedBMP = null;
                            for (Face face : faces) {
                                Rect bounds = face.getBoundingBox();
                                if ((bounds.left + bounds.width() <= rotatedBMP.getWidth()) &&
                                        (bounds.top + bounds.height() <= rotatedBMP.getHeight())
                                        && bounds.left > 0 && bounds.top > 0) {
                                    croppedBMP = Bitmap.createBitmap(rotatedBMP, bounds.left,
                                            bounds.top, bounds.width(), bounds.height());
                                    if (imageProxy.getImageInfo().getRotationDegrees() == 270) {
                                        croppedBMP = flipBitmap(croppedBMP);
                                    }
                                    String classification = ClassifyEmotion(croppedBMP);
                                    processFaceContourDetectionResult(faces, classification);
                                }
                            }
                            /*
                            Bitmap bmp = bmpImage.getBitmapInternal();
                            Bitmap rotatedBMP = rotateBitmap(bmp,
                                    imageProxy.getImageInfo().getRotationDegrees());
                            if (imageProxy.getImageInfo().getRotationDegrees() == 270) {
                                rotatedBMP = flipBitmap(rotatedBMP);
                                flipBox = true;
                            }

                            for (Face face : faces) {
                                Rect bounds = face.getBoundingBox();

                                Paint boxPaint;
                                boxPaint = new Paint();
                                boxPaint.setColor(Color.BLACK);
                                boxPaint.setStrokeWidth(10f);
                                boxPaint.setStyle(Paint.Style.STROKE);

                                Paint textPaint;
                                textPaint = new Paint();
                                textPaint.setColor(Color.BLACK);
                                textPaint.setTextSize((bounds.right - bounds.left)/5);
                                textPaint.setStrokeWidth(10f);
                                textPaint.setStyle(Paint.Style.FILL);

                                Bitmap croppedBMP = null;

                                if ((bounds.left + bounds.width() <= rotatedBMP.getWidth()) &&
                                        (bounds.top + bounds.height() <= rotatedBMP.getHeight())
                                        && bounds.left > 0 && bounds.top > 0) {
                                    croppedBMP = Bitmap.createBitmap(rotatedBMP, bounds.left,
                                            bounds.top, bounds.width(), bounds.height());
                                    String classification = ClassifyEmotion(croppedBMP);

                                    Canvas mCanvas = new Canvas(rotatedBMP);

                                }
                            }
                        }
                        */
                            }})
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            cameraXText.setText("Failed to run face detection");
                        }
                    });
            imageProxy.close();
        });

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        cameraProvider.bindToLifecycle(this, cameraSelector,
                imageAnalysis, preview);
    }

    // https://codelabs.developers.google.com/codelabs/mlkit-android#5
    private void processFaceContourDetectionResult(List<Face> faces, String classification) {

        mGraphicOverlay.clear();

        if (faces.size() == 0) {
            cameraXText.setText("No faces were found!");
            return;
        } else {
            cameraXText.setText("A face was detected!");
            for (int i = 0; i < faces.size(); ++i) {
                Face face = faces.get(i);

                if (face == null) {
                    return;
                }

                FaceContourGraphic faceGraphic = new FaceContourGraphic(
                        mGraphicOverlay, classification);

                if (frontCamera) {
                    mGraphicOverlay.setCameraInfo(480, 640,
                            CameraCharacteristics.LENS_FACING_BACK);
                } else {
                    mGraphicOverlay.setCameraInfo(480, 640,
                            CameraCharacteristics.LENS_FACING_FRONT);
                }

                mGraphicOverlay.add(faceGraphic);
                faceGraphic.updateFace(face);
            }
        }
    }

    // rotate the Bitmap returned by camera Intent
    // to normal (vertical) orientation
    public Bitmap rotateBitmap(Bitmap bmp, int newRotationAngle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(newRotationAngle);
        return Bitmap.createBitmap(bmp, 0, 0,
                bmp.getWidth(), bmp.getHeight(), matrix, true);
    }

    // flip image horizontally when using front (selfie)
    // camera for a more 'real' representation/view
    // from: https://shaikhhamadali.blogspot.com/2013/08/image-flipping-mirroring-in-imageview.html
    public Bitmap flipBitmap(Bitmap bmp) {
        Matrix matrix = new Matrix();
        matrix.preScale(-1.0f, 1.0f);
        return Bitmap.createBitmap(bmp, 0, 0,
                bmp.getWidth(), bmp.getHeight(), matrix, true);
    }

    // perform classification on detected face using the
    // custom trained CNN
    public String ClassifyEmotion (Bitmap detected_image) {
        try {
            Bitmap scaledResult = Bitmap.createScaledBitmap(detected_image,
                    48, 48, true);
            //imageView.setImageBitmap(scaledResult);
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(
                    this.getApplicationContext(), "emotion_cnn.tflite");

            Interpreter tflite = new Interpreter(tfliteModel);

            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new TransformToGrayscaleOp())
                    .build();
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(scaledResult);
            TensorImage newTensorImage = imageProcessor.process(tensorImage);
            TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(
                    new int[]{1, 7}, DataType.FLOAT32);

            tflite.run(newTensorImage.getBuffer(), probabilityBuffer.getBuffer());
            String classification = getClassification(probabilityBuffer.getFloatArray(), labels);
            return classification;
        } catch (IOException e) {
            System.out.println("Image classification failed!");
        } return null;
    }

    // print the biggest classification probability and its corresponding index
    private String getClassification(float[] floatArray, List<String> labels){
        DecimalFormat df = new DecimalFormat("0");
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

        String finalResult = labels.get(maxIndex) + ": " + df.format(maxValue*100) + "%";

        cameraXText.setText(finalResult);
        return finalResult;
    }

    // onClick function to allow user to return to MainActivity
    public void returnToHome(View v) {
        Intent returnIntent = new Intent(this, MainActivity.class);
        startActivity(returnIntent);
    }

    // flip between front and back camera (and vice versa)
    public void flipCamera(View v) {
        Intent intent = getIntent();
        intent.putExtra("from", "CameraXActivity");
        intent.putExtra("cameraChosen", frontCamera);
        finish();
        startActivity(intent);
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
