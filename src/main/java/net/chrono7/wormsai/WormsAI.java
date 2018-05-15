package net.chrono7.wormsai;

import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class WormsAI {

    public static final double GAMMA = 0.99;
    public static final boolean TRAINING_MODE = true;
    private static final int NET_DRIVE_AFTER_STEP = 500;
    private static final boolean USE_HUMAN_START = true;
    private static final int REFRESH_DELAY = 50;
    private static final int EXPLORE_STEPS = 7;
    private static final int MIN_STEP_FOR_NET = 50;
    private static final int DEATH_BUFFER = 15;
    private static final int TRAIN_EVERY_N_STEPS = 4;
    private static final int TRAIN_N_EXAMPLES = 32;
    private static final int CLONE_TARGET_EVERY_N_STEPS = 3000;
    private static final int PRINT_FREQUENCY = 10;
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END = 0.001;
    private static final double EPSILON_END_STEP = 1000;
    private static final double EPSILON_SLOPE = (EPSILON_END - EPSILON_START) / EPSILON_END_STEP;
    private static final boolean SAVE_NET = true;
    private static StateStore states = new StateStore(10_000);
    private static WebDriverExecutor webExe;
    private static int step = 0;
    private static int stepLastTrained = -1;
    private static int stepLastCloned = -1;
    private static int stepLastDeath = -1;
    private static long stepStartTime;
    private static MouseListener mouseListener = new MouseListener();
    private static NeuralNet4 net;
    private static Random rng = new Random();
    private static int exploreUntilStep = -1;
    private static int exploreInstruction = -1;

    public static void main(String[] args) {

//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        net = new NeuralNet4();

        registerMouseListener();

        webExe = new WebDriverExecutor();
        webExe.navigate();

        try {
            engage();
        } catch (Exception e) {
            e.printStackTrace();
        }
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

    private static void engage() throws InterruptedException, IOException {

        GameState prev = null;

        while (true) {

            BufferedImage capture = webExe.getScreenshot();

            if (webExe.testLoss()) { // Do training if we lost
                System.out.println("LOSS");

                states.applyToLastElements(s -> {
                    s.isTerminal = true;
                    s.reward = 0;
                }, DEATH_BUFFER);

                if (step > MIN_STEP_FOR_NET) {
                    if (step - stepLastCloned > CLONE_TARGET_EVERY_N_STEPS) {
                        net.cloneTarget();
                        stepLastCloned = step;
                    }
                    train(Math.min((step - stepLastDeath) * 4, 1000));
                }

                stepLastDeath = step;

                webExe.fixLoss();
                Thread.sleep(1500);
            }


            if (step % 10000 == 0 && SAVE_NET) {
                net.save(step);
            }

            INDArray arr;
            if (step > MIN_STEP_FOR_NET && step - stepLastTrained > 20 && step % 10 == 0) {
                Pair<Integer, INDArray> p = Vision4.processAndCountLargeBlobs(capture);
                if (p.getFirst() == 0) {
                    train(Math.min(states.getSize(), 30));
                }

                arr = p.getSecond();
            } else {
                arr = Vision4.process(capture);
            }

            GameState gs = new GameState(arr, step);

            int scorePre = (prev != null && !prev.isTerminal ? prev.score : 10);

            stepStartTime = System.currentTimeMillis();

            INDArray img = NeuralNet4.loader.asMatrix(capture);
            net.scaleImg(img);

            states.add(gs);

            drive(gs);

            delay(); // "do" the action

            int scorePost = webExe.getScore();

            gs.augment(scorePost, scorePost - scorePre);

            prev = gs;

            step++;

        }
    }

    private static void delay() throws InterruptedException {
        long desiredStop = stepStartTime + REFRESH_DELAY;

        long cur = System.currentTimeMillis();

        if (desiredStop > cur) {
            Thread.sleep(desiredStop - cur);
        }
    }

    private static void drive(GameState gs) {
        int actionIndex;
        boolean do_move = true;

        if (TRAINING_MODE) {

            if (step < NET_DRIVE_AFTER_STEP) {

                if (USE_HUMAN_START) {
                    actionIndex = Directions.getClosest(MouseInfo.getPointerInfo().getLocation(),
                            mouseListener.isMousePressed());
                    do_move = false;
                } else { // Uniform Random driver
                    actionIndex = getRandomAction();
                }

            } else { // Sometimes AI Driver

                if (step <= exploreUntilStep) { // If in a random period
                    actionIndex = getRandomAction();
                } else {
                    double epsilon = EPSILON_SLOPE * step + EPSILON_START;

                    if (Math.random() < epsilon) { // If starting a random period
                        actionIndex = getRandomAction();
                    } else { // AI selects
                        actionIndex = net.process(gs, step % PRINT_FREQUENCY == 0);
                    }
                }
            }
        } else {
            actionIndex = (step <= NeuralNet4.STACK_HEIGHT ? getRandomAction() :
                    net.process(gs, step % PRINT_FREQUENCY == 0));
        }


        if (do_move) {
            GameInstruction action = Directions.getInstruction(actionIndex);
            webExe.pointAdjusted(action.point);
            webExe.setBoost(action.boost);
        }

        gs.actionIndex = actionIndex;

        if (step % PRINT_FREQUENCY == 0) {
            System.out.println("step: " + step);
        }
    }

    private static int getRandomAction() {

        if (step > exploreUntilStep) {
            exploreUntilStep = step + EXPLORE_STEPS;
            exploreInstruction = Directions.randomIndex();
        }

        return exploreInstruction;
    }

    public static GameState getNext(GameState gs) {
        return states.get(gs.stepIndex + 1);
    }

    private static void train2() {
        ArrayList<GameState> examples = new ArrayList<>(TRAIN_N_EXAMPLES);

        for (int i = 0; i < TRAIN_N_EXAMPLES; i++) {
            examples.add(states.get(
                    Util.rand(states.getFirstIndex() + NeuralNet4.STACK_HEIGHT, states.getLastIndex() - DEATH_BUFFER)));
        }

        System.out.println("TRAINING on " + examples.size() + " examples");


        net.train(examples);
        System.out.println("TRAINING COMPLETE");
    }

    private static void train(int numExamples) {
        System.out.println("TRAINING");

        ArrayList<GameState> examples = new ArrayList<>(numExamples);

        for (int i = 0; i < numExamples; i++) {

            int rand = Util.rand(states.getFirstIndex() + NeuralNet4.STACK_HEIGHT,
                    states.getLastIndex() - DEATH_BUFFER);

            GameState gs = states.get(rand);

            if (!gs.isTerminal) {
                examples.add(gs);
            }
        }


        System.out.println("on " + examples.size() + " examples");
        net.train(examples);
        stepLastTrained = step;
        System.out.println("TRAINING COMPLETE");
    }

    /**
     * Gets the image for the state at the specified step, along with the previous (numStacked - 1) images stacked below.
     *
     * @param step       The step number for the topmost image state
     * @param numStacked The total number of image states to stack. Should be at least one.
     * @return The stacked image
     */
    public static INDArray getStackedImg(int step, int numStacked) {

        if (numStacked == 1) {
            return states.get(step).img;
        }

        INDArray[] arr = new INDArray[numStacked];

        for (int i = 0; i < numStacked; i++) {
            arr[i] = states.get(step - i).img;
        }

        return Nd4j.concat(1, arr);
    }

}