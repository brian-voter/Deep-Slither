package net.chrono7.deepslither;

import net.chrono7.deepslither.collections.CircularStore;
import net.chrono7.deepslither.state.GameState;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CircularStoreTest {

    private static CircularStore<GameState> states;

    @BeforeAll
    static void setUpAll () {
        states = new CircularStore<>(15_000);
    }

    @AfterEach
    void tearDown() {
        states.clear();
    }

    @Test
    void applyToLastElements() {
        states.add(new GameState(null), new GameState(null),
                new GameState(null), new GameState(null));

        assertEquals(states.size(), 4);

        states.applyToLastElements(s -> s.reward = 7, 2);
        assertEquals(states.get(0).reward, 0);
        assertEquals(states.get(1).reward, 0);
        assertEquals(states.get(2).reward, 7);
        assertEquals(states.get(3).reward, 7);

        states.applyToLastElements(s -> s.reward = 4, 1);
        assertEquals(states.get(0).reward, 0);
        assertEquals(states.get(1).reward, 0);
        assertEquals(states.get(2).reward, 7);
        assertEquals(states.get(3).reward, 4);

        states.applyToLastElements(s -> s.reward = 1, 4);
        assertEquals(states.get(0).reward, 1);
        assertEquals(states.get(1).reward, 1);
        assertEquals(states.get(2).reward, 1);
        assertEquals(states.get(3).reward, 1);

        states.applyToLastElements(s -> s.reward = 2, 0);
        assertEquals(states.get(0).reward, 1);
        assertEquals(states.get(1).reward, 1);
        assertEquals(states.get(2).reward, 1);
        assertEquals(states.get(3).reward, 1);
    }

    @Test
    void add() {
        assertEquals(states.size(), 0);
        states.add(new GameState(null));
        assertEquals(states.size(), 1);

        states.clear();
        assertEquals(states.size(), 0);

        for (int i = 0; i < states.capacity; i++) {
            states.add(new GameState(null));
        }
        assertEquals(states.size(), states.capacity);

        states.add(new GameState(null));
        assertEquals(states.size(), states.capacity);

    }

    @Test
    void clear() {
        for (int i = 0; i < 10; i++) {
            states.add(new GameState(null));
        }
        assertEquals(states.size(), 10);

        states.clear();

        assertEquals(states.size(), 0);
    }
}