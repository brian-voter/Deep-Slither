package net.chrono7.wormsai;

import java.awt.*;

public class GameInstruction {

    public final Point mouseLoc;
    public final boolean boost;

    public GameInstruction(Point mouseLoc, boolean boost) {

        this.mouseLoc = mouseLoc;
        this.boost = boost;
    }
}
