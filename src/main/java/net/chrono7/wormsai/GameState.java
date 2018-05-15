package net.chrono7.wormsai;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GameState {

    public INDArray img;
    public final long captureTime = System.currentTimeMillis();
    public final int stepIndex;
    public int actionIndex;
    public int score = Integer.MIN_VALUE;
    public int reward = Integer.MIN_VALUE;
    public boolean isTerminal = false;

    public GameState(INDArray img, int stepIndex) {
        this.img = img;
        this.stepIndex = stepIndex;
    }

    public GameState(INDArray img, int stepIndex, int actionIndex, int score, int reward) {
        this.img = img;
        this.actionIndex = actionIndex;
        this.score = score;
        this.reward = reward;
        this.stepIndex = stepIndex;
    }

    public GameState augment(int score, int reward) {
        if (this.score == Integer.MIN_VALUE) {
            this.score = score;
        }

        if (this.reward == Integer.MIN_VALUE) {
            this.reward = reward;
        }

        return this;
    }

    public GameState augment(int score, int reward, boolean isTerminal) {
        if (this.score == Integer.MIN_VALUE) {
            this.score = score;
        }

        if (this.reward == Integer.MIN_VALUE) {
            this.reward = reward;
        }

        this.isTerminal = isTerminal;

        return this;
    }

}
