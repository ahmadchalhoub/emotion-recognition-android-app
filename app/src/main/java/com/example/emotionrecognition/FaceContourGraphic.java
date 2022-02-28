package com.example.emotionrecognition;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.camera2.CameraCharacteristics;

import com.google.mlkit.vision.face.Face;

import java.util.List;

/** Graphic instance for rendering face contours graphic overlay view. */
public class FaceContourGraphic extends GraphicOverlay.Graphic {

    private static final float ID_TEXT_SIZE = 70.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;

    private final Paint facePositionPaint;
    private final Paint idPaint;
    private final Paint boxPaint;

    private final String classification;

    private volatile Face face;


    public FaceContourGraphic(GraphicOverlay overlay, String classification) {
        super(overlay);
        
        this.classification = classification;

        facePositionPaint = new Paint();
        facePositionPaint.setColor(Color.BLACK);

        idPaint = new Paint();
        idPaint.setColor(Color.BLACK);
        idPaint.setTextSize(ID_TEXT_SIZE);

        boxPaint = new Paint();
        boxPaint.setColor(Color.BLACK);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(BOX_STROKE_WIDTH);
    }

    /**
     * Updates the face instance from the detection of the most recent frame. Invalidates the relevant
     * portions of the overlay to trigger a redraw.
     */
    public void updateFace(Face face) {
        this.face = face;
        postInvalidate();
    }

    /** Draws the face annotations for position on the supplied canvas. */
    @Override
    public void draw(Canvas canvas) {
        Face face = this.face;
        if (face == null) {
            return;
        }

        // Scale and translate bounding box coordinates and text
        // position as needed
        float left = translateX(face.getBoundingBox().left);
        float right = translateX(face.getBoundingBox().right);
        float top = scaleY(face.getBoundingBox().top);
        float bottom = scaleY(face.getBoundingBox().bottom);

        // draw bounding box on detected face
        canvas.drawRect(left, top, right, bottom, boxPaint);

        // draw text on detected face
        if (left < right) {
            canvas.drawText(classification, left, bottom, idPaint);
        } else {
            canvas.drawText(classification, right, bottom, idPaint);
        }
    }
}
