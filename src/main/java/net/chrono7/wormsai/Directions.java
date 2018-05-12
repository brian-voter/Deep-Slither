package net.chrono7.wormsai;

import java.awt.*;
import java.util.Random;

public class Directions {

    private static final int[] xVals = new int[]{10, 314, 628, 942, 1256, 1570, 1870};
    private static final int[] yVals = new int[]{10, 160, 320, 480, 640, 800, 960};
    private static final GameInstruction[] instructions = generateInstructions();
    public static final int numInstructions = instructions.length;
    private static Random rng = new Random();

    public static int randomIndex() {
        return rng.nextInt(instructions.length);
    }

    public static GameInstruction getInstruction(int index) {
        return instructions[index];
    }

    private static GameInstruction[] generateInstructions() {
        GameInstruction[] instructions = new GameInstruction[2 * 2 * xVals.length + 2 * 2 * yVals.length];

        int i = 0;
        for (int x : xVals) {
            instructions[i++] = new GameInstruction(new Point(x, yVals[0]), false);
            instructions[i++] = new GameInstruction(new Point(x, yVals[0]), true);
            instructions[i++] = new GameInstruction(new Point(x, yVals[yVals.length - 1]), false);
            instructions[i++] = new GameInstruction(new Point(x, yVals[yVals.length - 1]), true);
        }

        for (int y : yVals) {
            instructions[i++] = new GameInstruction(new Point(xVals[0], y), false);
            instructions[i++] = new GameInstruction(new Point(xVals[0], y), true);
            instructions[i++] = new GameInstruction(new Point(xVals[xVals.length - 1], y), false);
            instructions[i++] = new GameInstruction(new Point(xVals[xVals.length - 1], y), true);
        }

        return instructions;
    }

    public static int getClosest(Point location, boolean boosting) {
        int minIdx = -1;
        double min = Integer.MAX_VALUE;

        for (int i = 0; i < instructions.length; i++) {

            if (boosting != instructions[i].boost) {
                continue;
            }

            double dist = instructions[i].point.distanceSq(location);
            if (dist < min) {
                min = dist;
                minIdx = i;
            }
        }

        return minIdx;
    }
}

