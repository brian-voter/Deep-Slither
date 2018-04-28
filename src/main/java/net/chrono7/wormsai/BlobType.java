package net.chrono7.wormsai;

import java.util.Arrays;

public enum BlobType {

    FOOD(0),
    PREY(1),
    WORM(2),
    UNKNOWN(-1);

    private int netLabel;

    BlobType(int netLabel) {
        this.netLabel = netLabel;
    }

    public static BlobType fromInt(int i) {
        return Arrays.stream(BlobType.values()).filter(b -> b.netLabel == i).findFirst().orElse(null);
    }

}
