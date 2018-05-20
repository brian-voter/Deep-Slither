package net.chrono7.deepslither;

import java.util.concurrent.ThreadLocalRandom;

/**
 * @author Brian Voter
 */
public class Util {

    public static int randInt(int minInclusive, int maxExclusive) {
        return ThreadLocalRandom.current().nextInt(minInclusive, maxExclusive);
    }

    public static double randDouble(double minInclusive, double maxExclusive) {
        return ThreadLocalRandom.current().nextDouble(minInclusive, maxExclusive);
    }

}
