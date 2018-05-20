package net.chrono7.wormsai.collections;


import java.util.Arrays;
import java.util.ListIterator;
import java.util.function.Consumer;

public class CircularStore<E> {

    public final int capacity;
    private CircularArrayList<E> list;

    public CircularStore(int capacity) {
        this.capacity = capacity;
        this.list = new CircularArrayList<>(capacity);
    }

    public int size() {
        return list.size();
    }

    /**
     * Adds a element to the store at the end of the list. The element at index 0 is
     * removed and returned if the list is full before the element is added.
     *
     * @param gs The element to add
     *
     * @return the element that was removed, or null if no element was removed.
     */
    public E add(E gs) {
        E removed = null;
        if (list.size() == capacity) {
            removed = list.remove(0);
        }

        list.add(gs);

        return removed;
    }

    /**
     * Adds elements to the store at the end of the list, in the order they are passed to this function. If all of the
     * elements will not fit in the list, elements are removed from the front of the list to make room.
     *
     * @param elements The elements to add
     */
    @SafeVarargs
    public final void add(E... elements) {

        int room = capacity - list.size();
        if (room < elements.length) {
            int removeN = elements.length - room;

            for (int i = 0; i < removeN; i++) {
                list.remove(0);
            }
        }

        list.addAll(Arrays.asList(elements));
    }

    public void clear() {
        list.clear();
    }

    public E get(int index) {
        return list.get(index);
    }

    /**
     * Applies the function op to the last nExamples elements in the list.
     *
     * @param op            The function to be applied
     * @param nLastElems The number of elements op will be applied to, starting from the end of the list and
     *                      approaching the head.
     * @throws IllegalArgumentException if nLastElems > size()
     */
    public void applyToLastElements(Consumer<E> op, int nLastElems) throws IllegalArgumentException {

        if (list.size() < nLastElems) {
            throw new IllegalArgumentException("nLastElems cannot be greater than list size!");
        }

        ListIterator<E> it = list.listIterator(list.size());

        E e;
        for (int i = 0; i < nLastElems; i++) {
            e = it.previous();

            op.accept(e);
        }
    }

}
