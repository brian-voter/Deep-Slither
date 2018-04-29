package net.chrono7.wormsai;

import org.opencv.core.Mat;

import java.awt.*;
import java.util.List;

public class VisionState {

    public VisionState(Mat mat, List<Blob> worms, List<Blob> food, List<Blob> prey, Blob self, long captureTime) {
        this.mat = mat;
        this.worms = worms;
        this.food = food;
        this.prey = prey;
        this.self = self;
        this.captureTime = captureTime;
    }

    public final Mat mat;
    public final List<Blob> worms;
    public final List<Blob> food;
    public List<Blob> prey;
    public final Blob self;
    public Mat hash;
    public String name;
    public double difference;
    public long captureTime;

    public int score;
    public int quality;
    public Point mouseLoc;
    public boolean boosting;

}
