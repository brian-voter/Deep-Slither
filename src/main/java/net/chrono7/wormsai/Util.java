package net.chrono7.wormsai;

import java.util.Random;

public class Util {

    private static Random rng = new Random();

    /**
     * @return a random integer between min [inclusive] and max [exclusive]
     */
    public static int rand(int min, int max) {
        return rng.nextInt(max - min) + min;
    }
}
