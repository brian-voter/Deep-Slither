package net.chrono7.wormsai;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GameState {

//    public opencv_core.Mat img;
    public INDArray arr;
    @Deprecated
    public final int stepIndex;
    public int actionIndex;
    public int score = Integer.MIN_VALUE;
    public int reward = Integer.MIN_VALUE;
    public boolean isTerminal = false;

    public GameState(INDArray arr, int stepIndex) {
        this.arr = arr;
        this.stepIndex = stepIndex;
    }

}
