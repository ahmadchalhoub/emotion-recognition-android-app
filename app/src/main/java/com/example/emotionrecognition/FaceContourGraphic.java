package com.example.emotionrecognition;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.mlkit.vision.face.Face;

/** Graphic instance for rendering face contours graphic overlay view. */
public class FaceContourGraphic extends GraphicOverlay.Graphic {

    private static final float ID_TEXT_SIZE = 70.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;

    private final Paint idPaint;
    private final Paint boxPaint;

    private volatile Face face;

    public FaceContourGraphic(GraphicOverlay overlay) {
        super(overlay);

        final int selectedColor = Color.BLACK;

        idPaint = new Paint();
        idPaint.setColor(selectedColor);
        idPaint.setTextSize(ID_TEXT_SIZE);

        boxPaint = new Paint();
        boxPaint.setColor(selectedColor);
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

        System.out.println("second left: " + face.getBoundingBox().left);
        System.out.println("second left: " + face.getBoundingBox().right);
        System.out.println("Detected face width: " + face.getBoundingBox().width());
        System.out.println("Detected face height: " + face.getBoundingBox().height());

        canvas.drawText("face", face.getBoundingBox().left, face.getBoundingBox().bottom, idPaint);
        canvas.drawRect(face.getBoundingBox().left+100, face.getBoundingBox().top,
                face.getBoundingBox().right, face.getBoundingBox().bottom, boxPaint);
    }
}
