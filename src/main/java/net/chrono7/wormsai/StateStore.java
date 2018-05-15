package net.chrono7.wormsai;

import java.util.TreeMap;
import java.util.function.Consumer;

public class StateStore {

    public StateStore(int capacity) {
        this.capacity = capacity;
    }

    public final int capacity;
    private TreeMap<Integer, GameState> states = new TreeMap<>();

    public int getSize() {
        return states.size();
    }

    public void add(GameState... add) {
        for (GameState state : add) {
            if (states.containsKey(state.stepIndex)) {
                throw new IllegalArgumentException("Step index " + state.stepIndex + " already in store");
            }

            if (states.size() == capacity) {
                states.pollFirstEntry();
            }

            states.put(state.stepIndex, state);
        }
    }

    public void clear() {
        states.clear();
    }

    public GameState get(int stepIndex) {
        return states.get(stepIndex);
    }

    public int getFirstIndex() {
        return states.firstKey();
    }

    public int getLastIndex() {
        return states.lastKey();
    }

    public void applyToLastElements(Consumer<GameState> op, int nLastExamples) {

        GameState state = states.lastEntry().getValue();
        for (int i = 0; i < nLastExamples; i++) {
            op.accept(state);

            if (i != nLastExamples - 1) {
                state = states.lowerEntry(state.stepIndex).getValue();
            }
        }
    }

}
