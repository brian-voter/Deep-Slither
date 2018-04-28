package net.chrono7.wormsai;

import org.jnativehook.mouse.NativeMouseEvent;
import org.jnativehook.mouse.NativeMouseInputListener;

import java.util.concurrent.atomic.AtomicBoolean;

public class MouseListener implements NativeMouseInputListener {

    private AtomicBoolean mousePressed = new AtomicBoolean(false);

    public boolean isMousePressed() {
        return mousePressed.get();
    }

    @Override
    public void nativeMouseClicked(NativeMouseEvent nativeEvent) {

    }

    @Override
    public void nativeMousePressed(NativeMouseEvent nativeEvent) {
        if (nativeEvent.getButton() == 1) {
            mousePressed.set(true);
        }
    }

    @Override
    public void nativeMouseReleased(NativeMouseEvent nativeEvent) {
        if (nativeEvent.getButton() == 1) {
            mousePressed.set(false);
        }
    }

    @Override
    public void nativeMouseMoved(NativeMouseEvent nativeEvent) {

    }

    @Override
    public void nativeMouseDragged(NativeMouseEvent nativeEvent) {

    }
}
