package com.example.emotionrecognition;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;


// display Camera preview in app
// CameraX Preview setup was obtained from:
// https://developer.android.com/training/camerax/preview
public class CameraXActivity extends MainActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camerax_activity);

        previewView = findViewById(R.id.previewView);

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

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview);
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
