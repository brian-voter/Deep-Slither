package net.chrono7.wormsai.state;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GameState {

    public INDArray before;
    public INDArray after;
    public int score = 0;
    public int reward = 0;
    public int actionIndex;
    public boolean isTerminal = false;

    public GameState(INDArray before) {
        this.before = before;
    }

}
