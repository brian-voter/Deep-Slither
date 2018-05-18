package net.chrono7.wormsai;

import java.awt.*;
import java.util.Random;

public class Directions {

    public static final boolean RELATIVE = false;
    private static final int[] relativeXVals = new int[]{-100, 0, 100};
    private static final int[] relativeYVals = new int[]{-100, 0, 100};
    private static final boolean[] boostVals = new boolean[]{false};
    private static final int[] xVals = new int[]{100, 628, 1256, 1800};
    private static final int[] yVals = new int[]{100, 320, 640, 900};
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
        if (RELATIVE) {
            GameInstruction[] instructions = new GameInstruction[relativeXVals.length * relativeYVals.length * boostVals.length];

            int i = 0;
            for (int x : relativeXVals) {
                for (int y : relativeYVals) {
                    for (boolean b : boostVals) {
                        instructions[i++] = new GameInstruction(new Point(x, y), b);
                    }
                }
            }

            return instructions;
        } else {
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
    }

    @Deprecated
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

