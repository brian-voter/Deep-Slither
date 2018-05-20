package net.chrono7.wormsai.collections;

/*
CREDIT: Adapted from https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
Original Python by Jaromír Janisch
Source: https://raw.githubusercontent.com/jaara/AI-blog/master/SumTree.py
 */

import org.nd4j.linalg.primitives.Pair;

public class SumTree<E> {

    // Here data = elem
    // and  p    = key

    public final int capacity;
    private final double[] tree;
    private final E[] data;
    private int write = 0;

    public SumTree(int capacity) {
        this.capacity = capacity;
        tree = new double[2 * capacity + 1];
        data = (E[]) new Object[capacity];
    }

    //key = p, elem = data
    private void propagate(int idx, double change) {
        int parent = (idx - 1) / 2;

        tree[parent] += change;

        if (parent != 0) {
            propagate(parent, change);
        }
    }

    private int retrieve(int idx, double s) {
        int left = 2 * idx + 1;
        int right = left + 1;

        if (left >= tree.length) {
            return idx;
        }

        if (s <= tree[left]) {
            return retrieve(left, s);
        } else {
            return retrieve(right, s - tree[left]);
        }
    }

    public double total() {
        return tree[0];
    }

    //key = p, elem = data
    public void add(double key, E elem) {
        int idx = write + capacity - 1;

        data[write] = elem;
        update(idx, key);

        write += 1;
        if (write >= capacity) {
            write = 0;
        }
    }

    public void update(int idx, double key) {
        double change = key - tree[idx];

        tree[idx] = key;
        propagate(idx, change);
    }

    public Pair<Integer, E> get(double s) {
        int idx = retrieve(0, s);
        int dataIdx = idx - capacity + 1;


        //TODO: return more?
        return new Pair<>(idx, data[dataIdx]);
    }


}
