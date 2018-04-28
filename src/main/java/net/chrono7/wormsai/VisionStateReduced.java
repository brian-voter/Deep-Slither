package net.chrono7.wormsai;

import org.opencv.core.Mat;

import java.awt.*;
import java.util.Random;
import java.util.UUID;

public class VisionStateReduced {

    private static Random random = new Random();

    /**
     * Creates a new VisionStateReduced based on vs and with the SAME UUID as vs.
     * @param vs The VisionState to copy
     */
    public static VisionStateReduced clone(VisionStateReduced vs) {
        VisionStateReduced newVS = new VisionStateReduced(vs);
        newVS.uuid = vs.uuid;
        return newVS;
    }

    /**
     * Creates a new VisionStateReduced based on vs, but with a new UUID
     * @param vs The VisionState to copy
     */
    public VisionStateReduced(VisionState vs) {
        this.hash = vs.hash.clone();
        this.score = vs.score;
        this.quality = vs.quality;
        if (vs.mouseLoc != null) {
            this.mouseLoc = (Point) vs.mouseLoc.clone();
        }
        this.boosting = vs.boosting;
        this.uuid = new UUID(System.currentTimeMillis(), random.nextLong());
    }

    /**
     * Creates a new VisionStateReduced based on vs, but with a new UUID
     * @param vs The VisionStateReduced to copy
     */
    public VisionStateReduced(VisionStateReduced vs) {
        this.hash = vs.hash.clone();
        this.score = vs.score;
        this.quality = vs.quality;
        if (vs.mouseLoc != null) {
            this.mouseLoc = (Point) vs.mouseLoc.clone();
        }
        this.boosting = vs.boosting;
        this.uuid = new UUID(System.currentTimeMillis(), random.nextLong());
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof VisionStateReduced)) {
            return false;
        }

        VisionStateReduced vsr = (VisionStateReduced) other;

        return this.uuid.equals(vsr.uuid);
    }

    private UUID uuid;
    public final Mat hash;
    public final int score;
    public int quality;
    public Point mouseLoc;
    public final boolean boosting;
}
