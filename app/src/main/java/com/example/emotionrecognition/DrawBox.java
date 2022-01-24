package com.example.emotionrecognition;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.View;

public class DrawBox extends View {

    Rect rectInput;
    Paint boxPaint;

    public DrawBox(Context context, Rect rect) {
        super(context);
        this.rectInput = rect;

        boxPaint = new Paint();
        boxPaint.setColor(Color.WHITE);
        boxPaint.setStrokeWidth(10f);
        boxPaint.setStyle(Paint.Style.STROKE);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawRect(rectInput.left, rectInput.top, rectInput.right, rectInput.bottom, boxPaint);
    }

}
