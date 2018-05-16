package net.chrono7.wormsai;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GameState {

    public INDArray img;
    public final int stepIndex;
    public int actionIndex;
    public int score = Integer.MIN_VALUE;
    public int reward = Integer.MIN_VALUE;
    public boolean isTerminal = false;

    public GameState(INDArray img, int stepIndex) {
        this.img = img;
        this.stepIndex = stepIndex;
    }

}
