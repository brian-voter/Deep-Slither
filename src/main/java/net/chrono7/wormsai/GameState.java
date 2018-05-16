package net.chrono7.wormsai;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Frame;

public class GameState {

    public opencv_core.Mat img;
    public final int stepIndex;
    public int actionIndex;
    public int score = Integer.MIN_VALUE;
    public int reward = Integer.MIN_VALUE;
    public boolean isTerminal = false;

    public GameState(Frame img, int stepIndex) {
        this.img = Vision4.shrink(img);
        this.stepIndex = stepIndex;
    }

}
