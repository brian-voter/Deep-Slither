package net.chrono7.wormsai;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

public class Blob {

    public MatOfPoint contour;
    public MatOfPoint2f mp2f;
    public double distanceFromCenter;
    public double area;
    public BlobType blobType;
    public Mat mat;

    public Blob(MatOfPoint contour, MatOfPoint2f mp2f, double distanceFromCenter, double area, BlobType blobType) {
        this.contour = contour;
        this.mp2f = mp2f;
        this.distanceFromCenter = distanceFromCenter;
        this.area = area;
        this.blobType = blobType;
    }
}
