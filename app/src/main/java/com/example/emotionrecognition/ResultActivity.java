package com.example.emotionrecognition;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
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
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

public class ResultActivity extends MainActivity {

    List<String> labels = Arrays.asList("Angry", "Disgust", "Fear", "Happy",
            "Sad", "Surprise", "Neutral");

    private TextView resultText;
    private ImageView resultImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.result_activity);

        resultText = findViewById(R.id.resultText);
        resultImage = findViewById(R.id.resultImage);

        Intent resultIntent = getIntent();
        Bundle b = resultIntent.getExtras();
        Bitmap rotatedBitmap = (Bitmap) b.get("rotatedPhoto");
        try {
            DetectFace(rotatedBitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }

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
            resultText.setText("No faces were detected in the image. Try again!");
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

            Rect rectCrop = new Rect(highestX, highestY, highestWidth, highestHeight);
            Mat croppedImage = new Mat(img, rectCrop);

            Bitmap croppedResult;
            croppedResult = Bitmap.createBitmap(highestWidth, highestHeight,
                    Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedImage, croppedResult);
            resultImage.setImageBitmap(croppedResult);

            Bitmap scaledResult = Bitmap.createScaledBitmap(croppedResult,
                    48, 48, true);
            ClassifyEmotion(scaledResult);
        }
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
        resultText.setText(finalResult);
    }
}
