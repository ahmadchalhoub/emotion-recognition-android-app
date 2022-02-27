package com.example.emotionrecognition;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.mlkit.vision.face.Face;

import java.util.List;

/** Graphic instance for rendering face contours graphic overlay view. */
public class FaceContourGraphic extends GraphicOverlay.Graphic {

    private static final float ID_TEXT_SIZE = 70.0f;
    private static final float ID_Y_OFFSET = 80.0f;
    private static final float ID_X_OFFSET = -70.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;

    private final Paint facePositionPaint;
    private final Paint idPaint;
    private final Paint boxPaint;

    private volatile Face face;

    public FaceContourGraphic(GraphicOverlay overlay) {
        super(overlay);

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

        // Draw text on the face
        float x = translateX(face.getBoundingBox().centerX());
        //System.out.println("centerX = " + face.getBoundingBox().centerX());
        //System.out.println("translated centerX = " + translateX(face.getBoundingBox().centerX()));

        float y = translateY(face.getBoundingBox().centerY());
        //System.out.println("centerY = " + face.getBoundingBox().centerY());
        //System.out.println("translated centerY = " + translateY(face.getBoundingBox().centerY()));
        //canvas.drawText("id: " + face.getTrackingId(), x + ID_X_OFFSET,
        //        y + ID_Y_OFFSET, idPaint);

        // Draws a bounding box around the face.
        //float xOffset = scaleX(face.getBoundingBox().width() / 2.0f);
        //float yOffset = scaleY(face.getBoundingBox().height() / 2.0f);
        //float left = x - xOffset;
        //float top = y - yOffset;
        //float right = x + xOffset;
        //float bottom = y + yOffset;

        float left = scaleX(face.getBoundingBox().left);
        float right = scaleX(face.getBoundingBox().right);
        float top = scaleY(face.getBoundingBox().top);
        float bottom = scaleY(face.getBoundingBox().bottom);

        canvas.drawRect(left, top, right, bottom, boxPaint);

        System.out.println("left = " + face.getBoundingBox().left);
        System.out.println("top = " + face.getBoundingBox().top);
        System.out.println("right = " + face.getBoundingBox().right);
        System.out.println("bottom = " + face.getBoundingBox().bottom);
    }
}
