package net.chrono7.wormsai;

import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.img_hash.ColorMomentHash;
import org.opencv.img_hash.ImgHashBase;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import javax.swing.*;
import java.awt.*;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class WormsAI {

    private static final int REFRESH_DELAY = 100;
    public static final String CAPTURE_DIRECTORY = "D:\\Documents\\wormImages\\";
    private static final int SCORE_WINDOW = 30;
    private static ArrayList<VisionStateReduced> states = new ArrayList<>();
    private static Vision3 vision;
    private static ImgHashBase hashAlgo;
    private static WebDriverExecutor webExe;
    private static int steps = 0;
    private static MouseListener mouseListener = new MouseListener();
    private static JFrame frame;
    private static JLabel imageLabel;

    public static void main(String[] args) throws AWTException {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        try {
            vision = new Vision3();
        } catch (IOException e) {
            e.printStackTrace();
        }

        registerMouseListener();

        webExe = new WebDriverExecutor();
        webExe.navigate();

        captureLoop();
    }

    private static void registerMouseListener() {
        try {
            GlobalScreen.registerNativeHook();
        } catch (NativeHookException e) {
            e.printStackTrace();
        }

        GlobalScreen.addNativeMouseListener(mouseListener);
        Logger.getLogger(GlobalScreen.class.getPackage().getName()).setLevel(Level.SEVERE);
    }

    private static void captureLoop() {

        hashAlgo = ColorMomentHash.create();

        long stepStartTime;

        try {
            while (states.size() < 400) {
                stepStartTime = System.currentTimeMillis();
                steps++;
                BufferedImage capture = webExe.getScreenshot();

//                String name = String.valueOf(System.currentTimeMillis());

                VisionState vs = vision.process(capture);

                setupFrame(vs);

                imageLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(vs.mat)));

//                System.out.println(we.getScore());

                Mat hashed = new Mat(vs.mat.rows(), vs.mat.cols(), CvType.CV_8UC3, Scalar.all(0));
                hashAlgo.compute(vs.mat, hashed);

                vs.mouseLoc = MouseInfo.getPointerInfo().getLocation();
                vs.boosting = mouseListener.isMousePressed();

//                vs.name = name;
                vs.hash = hashed;

                vs.score = webExe.getScore();

                updateScore(vs);

//                if (states.size() == 100) {
//                    target = vs;
//
//                    states.forEach(s -> s.difference = hashAlgo.compare(target.hash, s.hash));
//
//                    states.sort(Comparator.comparingDouble(s -> s.difference));
//
//                    states.forEach(s -> System.out.println("Name: " + s.name + " | diff: " + s.difference));
//
//                    Imgcodecs.imwrite(CAPTURE_DIRECTORY + "target.png", target.mat);
//                    Imgcodecs.imwrite(CAPTURE_DIRECTORY + "nearest.png", states.get(0).mat);
//
//                    break;
//                } else {
//                    states.add(vs);
//                }

                delayStep(stepStartTime);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        engageAI();

    }

    private static void setupFrame(VisionState vs) {
        if (frame == null) {
            frame = new JFrame("WormsAI");
            frame.setSize(vs.mat.width(), vs.mat.height());
            frame.setLocation(-1900, 500);
            frame.setVisible(true);
            imageLabel = new JLabel(new ImageIcon(HighGui.toBufferedImage(vs.mat)));
            frame.add(imageLabel);
        }
    }

    private static void delayStep(long stepStartTime) throws InterruptedException {
        long stepFinishTime;
        stepFinishTime = System.currentTimeMillis();

        if (stepStartTime + REFRESH_DELAY > stepFinishTime) {
            Thread.sleep(stepStartTime + REFRESH_DELAY - stepFinishTime);
        }
    }

    private static void engageAI() {

        System.out.println("AUTO PILOT ENGAGE!!!\n\n");

        long stepStartTime;

        try {
            while (true) {

                stepStartTime = System.currentTimeMillis();

                if (steps % 4 == 0) {
                    webExe.fixLoss();
                }

                if (steps % 100 == 0) { //Do state reduction
                    doStateReduction();
                }

                BufferedImage capture = webExe.getScreenshot();

                VisionState vs = vision.process(capture);

                imageLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(vs.mat)));

                Mat hashed = new Mat(new Size(vs.mat.rows(), vs.mat.cols()), CvType.CV_8UC3, Scalar.all(0));
                hashAlgo.compute(vs.mat, hashed);
                vs.score = webExe.getScore();
                vs.hash = hashed;

                Point point = null;
                boolean boost = false;

                if (vs.worms.size() == 0 && vs.prey.size() == 0) {
                    if (vs.food.size() > 0) {
                        try {
                            Blob max = vs.food.stream().max(Comparator.comparingDouble(c ->
                                    c.area * (1.0 / c.distanceFromCenter))).get();

                            Moments moments = Imgproc.moments(max.contour);

                            org.openqa.selenium.Point tl = webExe.getTopLeftPoint();
                            point = new Point((int) (moments.m10 / moments.m00) + tl.x,
                                    (int) (moments.m01 / moments.m00) + tl.y);
                        } catch (NullPointerException e) {
                            System.out.println("null max " + steps);
                        }
                    }
                } else {
                    VisionStateReduced best = states.stream().sorted(Comparator.comparingDouble(s ->
                            hashAlgo.compare(hashed, s.hash))).limit(10).sorted(Comparator.comparingDouble(s -> s.quality))
                            .collect(Collectors.toList()).get(0);

                    point = best.mouseLoc;
                    boost = best.boosting;
                }

                if (point != null) {
                    webExe.point(point);
                }

                webExe.setBoost(boost);

                vs.mouseLoc = point;
                vs.boosting = boost;

                updateScore(vs);

                steps++;
                delayStep(stepStartTime);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void doStateReduction() {
        int sizeBefore = states.size();
        states.removeIf(s -> Math.abs(s.quality) < 5);
        System.out.println("Pruned: " + (sizeBefore - states.size()));

        sizeBefore = states.size();

        ArrayList<VisionStateReduced> combined = new ArrayList<>(states.size());

        states.forEach(s -> combined.add(VisionStateReduced.clone(s)));

        for (int i = 0; i < states.size(); i++) {
            for (int j = i + 1; j < states.size(); j++) {
                if (hashAlgo.compare(states.get(i).hash, states.get(j).hash) < 5) {
                    VisionStateReduced newVSR = states.get(i).quality > states.get(j).quality ?
                            VisionStateReduced.clone(states.get(i)) : VisionStateReduced.clone(states.get(j));
//                    newVSR.quality = (states.get(i).quality + states.get(j).quality) / 2;
                    combined.add(newVSR);
                    combined.remove(states.get(i));
                    combined.remove(states.get(j));
                }
            }
        }

        states = combined;
        System.out.println("Combined: " + (sizeBefore - states.size()));
        System.out.println("States: " + states.size());
    }

    private static void updateScore(VisionState vs) {
        if (states.size() > SCORE_WINDOW) {

            int statesAgo = 1;

            int changeInScore = vs.score - states.get(states.size() - 1).score;

            for (int i = states.size() - 1; i >= states.size() - SCORE_WINDOW; i--) {
                states.get(i).quality += changeInScore * ((SCORE_WINDOW - statesAgo + 1.0) / SCORE_WINDOW);
                statesAgo++;
            }
        }

        states.add(new VisionStateReduced(vs));

//        printScores();
    }

    private static void printScores() {

        final int printWindow = 20;

        System.out.println("SCORE: " + states.get(states.size() - 1).score);

        if (states.size() > printWindow) {
            int startIdx = states.size() - printWindow;
            StringBuilder scoreString = new StringBuilder(String.valueOf(states.get(startIdx).quality));
            for (int i = startIdx + 1; i < states.size(); i++) {
                scoreString.append(", ").append(states.get(i).quality);
            }
            System.out.println(scoreString.toString() + "\n");
        }

    }

}