package net.chrono7.deepslither.state;


import net.chrono7.deepslither.Util;
import net.chrono7.deepslither.collections.CircularStore;
import net.chrono7.deepslither.collections.SumTree;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;

/**
 * CREDIT:
 * Inspired by and adapted from:
 * https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
 */
public class StateStore {

    private static final double PER_e = 0.6;
    private static final double PER_a = 0.01;
    private SumTree<GameState> states;
    private CircularStore<GameState> deathBuffer;
    private double maxPriority = 1;

    public StateStore(int capacity, int deathBufferSize) {
        this.states = new SumTree<>(capacity);
        this.deathBuffer = new CircularStore<>(deathBufferSize);
    }

    public double getPriority(double error) {
        return Math.pow((error + PER_e), PER_a);
    }

    public void push(GameState state) {
        GameState popped = deathBuffer.add(state);

        if (popped != null) {
            states.add(maxPriority, popped);
        }
    }

    public void notifyDeath() {
        for (int i = 0; i < deathBuffer.size(); i++) {
            GameState gs = deathBuffer.get(i);
            gs.isTerminal = true;
            gs.reward = 0;
        }
    }

    //TODO: fix to return exactly numExamples (note: can hang if a while loop is used in the for loop)
    public ArrayList<Pair<Integer, GameState>> sample(int numExamples) {
        ArrayList<Pair<Integer, GameState>> examples = new ArrayList<>();

        double segment = states.total() / numExamples;

        for (int i = 0; i < numExamples; i++) {

            double a = segment * i;
            double b = segment * (i + 1);

            double s = Util.randDouble(a, b);

            Pair<Integer, GameState> pair = states.get(s);

            if (!pair.getSecond().isTerminal) {
                examples.add(pair);
            }
        }

        return examples;
    }

    public void update(int idx, double error) {
        double priority = getPriority(error);
        if (priority > maxPriority) {
            maxPriority = priority;
        }

        states.update(idx, priority);
    }

}
