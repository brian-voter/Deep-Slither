package net.chrono7.wormsai;

import java.awt.*;

public class GameInstruction {

    public final Point point;
    public final boolean boost;

    public GameInstruction(Point point, boolean boost) {
        this.point = point;
        this.boost = boost;
    }

    @Override
    public String toString() {
        return point.toString() + " boost: " + boost;
    }
}
