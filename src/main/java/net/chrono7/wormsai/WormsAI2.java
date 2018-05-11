package net.chrono7.wormsai;

import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class WormsAI2 {

    public static final String CAPTURE_DIRECTORY = "D:\\Documents\\wormImages\\";
    private static final int REFRESH_DELAY = 100;
    private static final int SCORE_WINDOW = 5;
    public static String STATES_FEATURES_FILE = "D:\\Documents\\wormsStates\\features.csv";
    public static String STATES_LABELS_FILE = "D:\\Documents\\wormsStates\\labels.csv";
    public static String STATES_TIMESTAMPS_FILE = "D:\\Documents\\wormsStates\\timestamps.csv";
    public static String STATES_CAPTURE_DIR = "D:\\Documents\\wormsStates\\images\\";
    private static LinkedList<GameState> states = new LinkedList<>();
    //    private static NeuralNet2 net2;
//    private static ImgHashBase hashAlgo;
    private static WebDriverExecutor webExe;
    private static int steps = 0;
    private static MouseListener mouseListener = new MouseListener();
    private static JFrame frame;
    private static JLabel imageLabel;
    private static FileWriter featuresWriter;
    private static FileWriter labelsWriter;
    private static FileWriter timeStampWriter;

    public static void main(String[] args) throws AWTException {

//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        try {
//            net2 = new NeuralNet2();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        File dir = new File(STATES_CAPTURE_DIR);
        dir.mkdirs();

        registerMouseListener();

        webExe = new WebDriverExecutor();
        webExe.navigate();

        try {
            featuresWriter = new FileWriter(STATES_FEATURES_FILE, true);
            labelsWriter = new FileWriter(STATES_LABELS_FILE, true);
            timeStampWriter = new FileWriter(STATES_TIMESTAMPS_FILE, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        captureLoop();
//        engageAI();
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

        long stepStartTime;

        try {
            while (steps < 2000) {
                if (webExe.fixLoss()) {
                    handleLoss();
                }

                System.out.println("step: " + steps);
                stepStartTime = System.currentTimeMillis();
                steps++;
                BufferedImage capture = webExe.getScreenshot();

                GameState gs = new GameState(capture);

                gs.mouseLoc = MouseInfo.getPointerInfo().getLocation();
                gs.boosting = mouseListener.isMousePressed();

                gs.score = webExe.getScore();

                updateScore(gs);

                delayStep(stepStartTime);

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

//        engageAI();

    }

    private static void handleLoss() {
        for (GameState cur : states) {
            cur.quality = Integer.MIN_VALUE;
        }
    }
//
//    private static void setupFrame(VisionState vs) {
//        if (frame == null) {
//            frame = new JFrame("WormsAI");
//            frame.setSize(vs.mat.width(), vs.mat.height());
//            frame.setLocation(-1900, 500);
//            frame.setVisible(true);
//            imageLabel = new JLabel(new ImageIcon(HighGui.toBufferedImage(vs.mat)));
//            frame.add(imageLabel);
//        }
//    }

    private static void delayStep(long stepStartTime) throws InterruptedException {
        long stepFinishTime;
        stepFinishTime = System.currentTimeMillis();

        if (stepStartTime + REFRESH_DELAY > stepFinishTime) {
            Thread.sleep(stepStartTime + REFRESH_DELAY - stepFinishTime);
        }
    }

//    private static void engageAI() {
//
//        System.out.println("AUTO PILOT ENGAGE!!!\n\n");
//
//        long stepStartTime;
//
//        try {
//            while (true) {
//
//                stepStartTime = System.currentTimeMillis();
//
//                if (steps % 4 == 0) {
//                    webExe.fixLoss();
//                }
//
////                if (steps % 100 == 0) { //Do state reduction
////                    doStateReduction();
////                }
//
//                BufferedImage capture = webExe.getScreenshot();
//
//                GameInstruction instr = net2.process(capture);
//
//                GameState gs = new GameState(capture);
//
//                Point point = null;
//                boolean boost = false;
//
//
//                point = instr.mouseLoc;
//                boost = instr.boost;
//
//                System.out.println(point);
//
//                if (point != null) {
//                    webExe.point(point);
//                }
//
//                webExe.setBoost(boost);
//
//                gs.mouseLoc = point;
//                gs.boosting = boost;
//
//                gs.score = webExe.getScore();
//
//                updateScore(gs);
//
//                steps++;
//                delayStep(stepStartTime);
//
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

    private static void doStateReduction() {
        int sizeBefore = states.size();
        states.removeIf(s -> Math.abs(s.quality) < 5);
        System.out.println("Pruned: " + (sizeBefore - states.size()));

        sizeBefore = states.size();

//        ArrayList<VisionStateReduced> combined = new ArrayList<>(states.size());
//
//        states.forEach(s -> combined.add(VisionStateReduced.clone(s)));
//
//        for (int i = 0; i < states.size(); i++) {
//            for (int j = i + 1; j < states.size(); j++) {
//                if (hashAlgo.compare(states.get(i).hash, states.get(j).hash) < 5) {
//                    VisionStateReduced newVSR = states.get(i).quality > states.get(j).quality ?
//                            VisionStateReduced.clone(states.get(i)) : VisionStateReduced.clone(states.get(j));
////                    newVSR.quality = (states.get(i).quality + states.get(j).quality) / 2;
//                    combined.add(newVSR);
//                    combined.remove(states.get(i));
//                    combined.remove(states.get(j));
//                }
//            }
//        }
//
//        states = combined;
//        System.out.println("Combined: " + (sizeBefore - states.size()));
        System.out.println("States: " + states.size());
    }

    private static void updateScore(GameState gs) {
        if (states.size() > SCORE_WINDOW) {

            int statesAgo = 1;

            int changeInScore = gs.score - states.getFirst().score;

            for (GameState cur : states) {
                if (cur.quality != Integer.MIN_VALUE) {
                    cur.quality += changeInScore * ((SCORE_WINDOW - statesAgo + 1.0) / SCORE_WINDOW);
                }
                statesAgo++;
            }

            GameState last = states.removeLast();

            writeStateToDisk(last);

//            try {
//                net2.trainState(last);
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
        }

        states.addFirst(gs);
    }

    private static void writeStateToDisk(GameState gs) {

        try {

            //FORMAT: time x y boosting quality

            featuresWriter.write(gs.mouseLoc.x + "," + gs.mouseLoc.y + "\n");
            featuresWriter.flush();
            labelsWriter.write((gs.quality == Integer.MIN_VALUE ? -100 : gs.quality) + "\n");
            labelsWriter.flush();
            timeStampWriter.write(gs.captureTime + "\n");
            timeStampWriter.flush();

            ImageIO.write(gs.img, "png", new File(STATES_CAPTURE_DIR + gs.captureTime + ".png"));

        } catch (IOException e) {
            e.printStackTrace();
        }

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