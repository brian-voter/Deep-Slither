package net.chrono7.wormsai;

import java.awt.*;
import java.awt.image.BufferedImage;

public class GameState {

    public final BufferedImage img;
    public Point mouseLoc;
    public boolean boosting;
    public int score;
    public int quality;
    public final long captureTime = System.currentTimeMillis();

    public GameState(BufferedImage img) {
        this.img = img;
    }

    public GameState(BufferedImage img, Point mouseLoc, boolean boosting, int score) {

        this.img = img;
        this.mouseLoc = mouseLoc;
        this.boosting = boosting;
        this.score = score;
    }

}
