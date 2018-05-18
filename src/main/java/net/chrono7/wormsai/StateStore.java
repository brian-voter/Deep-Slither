package net.chrono7.wormsai;


import java.util.Arrays;
import java.util.ListIterator;
import java.util.function.Consumer;

public class StateStore {

    public final int capacity;
    //    private TreeMap<Integer, GameState> states = new TreeMap<>();
    private CircularArrayList<GameState> states;

    public StateStore(int capacity) {
        this.capacity = capacity;
        this.states = new CircularArrayList<>(capacity);
    }

    public int size() {
        return states.size();
    }

    /**
     * Adds a state to the store at the end of the list. The state at index 0 is
     * removed if the list is full before the state is added.
     *
     * @param gs The state to add
     */
    public void add(GameState gs) {
        if (states.size() == capacity) {
            states.remove(0);
        }

        states.add(gs);
    }

    /**
     * Adds states to the store at the end of the list, in the order they are passed to this function. If all of the
     * states will not fit in the list, states are removed from the front of the list to make room.
     *
     * @param states The states to add
     */
    public void add(GameState... states) {

        int room = capacity - this.states.size();
        if (room < states.length) {
            int removeN = states.length - room;

            for (int i = 0; i < removeN; i++) {
                this.states.remove(0);
            }
        }

        this.states.addAll(Arrays.asList(states));
    }

    public void clear() {
        states.clear();
    }

    public GameState get(int stepIndex) {
        return states.get(stepIndex);
    }

    /**
     * Applies the function op to the last nExamples states in the list.
     *
     * @param op            The function to be applied
     * @param nLastExamples The number of states op will be applied to, starting from the end of the list and
     *                      approaching the head.
     * @throws IllegalArgumentException if nLastExamples > size()
     */
    public void applyToLastElements(Consumer<GameState> op, int nLastExamples) throws IllegalArgumentException {

        if (states.size() < nLastExamples) {
            throw new IllegalArgumentException("nLastExamples cannot be greater than list size!");
        }

        ListIterator<GameState> it = states.listIterator(states.size());

        GameState state;
        for (int i = 0; i < nLastExamples; i++) {
            state = it.previous();

            op.accept(state);
        }
    }

}
