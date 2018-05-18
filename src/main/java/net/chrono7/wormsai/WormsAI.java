package net.chrono7.wormsai;

import org.bytedeco.javacv.Frame;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class WormsAI {

    public static final double GAMMA = 0.99;
    public static final boolean TRAINING_MODE = true;
    //    private static final int NET_DRIVE_AFTER_STEP = 10_000;
    private static final int NET_DRIVE_AFTER_STEP = 200;
    private static final boolean USE_HUMAN_START = false;
    private static final int REFRESH_DELAY = 50;
    private static final int EXPLORE_STEPS = 5;
    //    private static final int MIN_STEP_FOR_NET = 5000; // MAKE DIVISIBLE BY TRAIN_EVERY_N_STEPS
    private static final int MIN_STEP_FOR_NET = 100; // MAKE DIVISIBLE BY TRAIN_EVERY_N_STEPS
    private static final int DEATH_BUFFER = 20;
    private static final int TRAIN_EVERY_N_STEPS = 5;
    private static final int TRAIN_N_EXAMPLES = 35;
    private static final int CLONE_TARGET_EVERY_N_STEPS = 750;
    private static final int PRINT_FREQUENCY = 10;
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END = 0.001;
    //    private static final double EPSILON_END_STEP = 75_000;
    private static final double EPSILON_END_STEP = 500;
    private static final double EPSILON_SLOPE = (EPSILON_END - EPSILON_START) / EPSILON_END_STEP;
    private static final boolean SAVE_NET = true;
    public static MouseListener mouseListener = new MouseListener();
    private static StateStore states = new StateStore(20_000);
    private static WebDriverExecutor webExe;
    private static int step = 0;
    private static int stepLastTrained = -1;
    private static int stepLastCloned = -1;
    private static int stepLastDeath = -1;
    private static long stepStartTime;
    private static NeuralNet4 net;
    private static Random rng = new Random();
    private static int exploreUntilStep = -1;
    private static int exploreInstruction = -1;

    public static void main(String[] args) {

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

            stepStartTime = System.currentTimeMillis();
            Frame capture = webExe.getScreenshot();

            if (webExe.testLoss()) { // Do training if we lost
                System.out.println("LOSS");

                states.applyToLastElements(s -> {
                    s.isTerminal = true;
                    s.reward = 0;
                }, DEATH_BUFFER);

                stepLastDeath = step;

                webExe.fixLoss();
                Thread.sleep(1500);
            }

            if (step % 10000 == 0 && SAVE_NET) {
                net.save(step);
            }

            GameState gs = new GameState(capture, step);
            states.add(gs);

            drive(gs);

            if (step >= MIN_STEP_FOR_NET && step % TRAIN_EVERY_N_STEPS == 0) {
                if (step - stepLastCloned > CLONE_TARGET_EVERY_N_STEPS || step == MIN_STEP_FOR_NET) {
                    net.cloneTarget();
                    stepLastCloned = step;
                }
                train(Math.min(states.size(), TRAIN_N_EXAMPLES));
            }

            int scorePre = (prev != null && !prev.isTerminal ? prev.score : 10);

            delay(); // "do" the action

            int scorePost = webExe.getScore();

            if (step - stepLastDeath <= 25) {
                gs.score = 10;
                gs.reward = 0;
            } else {
                gs.score = scorePost;
                gs.reward = scorePost - scorePre;
            }

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
                    //TODO: re-enable human start?
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
                        actionIndex = net.predictBestAction(states.size() - 1, step % PRINT_FREQUENCY == 0);
                    }
                }
            }
        } else {
            actionIndex = (step <= NeuralNet4.STACK_HEIGHT ? getRandomAction() :
                    net.predictBestAction(states.size(), step % PRINT_FREQUENCY == 0));
        }


        if (do_move) {
            GameInstruction action = Directions.getInstruction(actionIndex);
            webExe.act(action);
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

    public static GameState getState(int index) {
        return states.get(index);
    }

    private static void train(int numExamples) {
        System.out.println("TRAINING");

        ArrayList<GameState> examples = new ArrayList<>(numExamples);
        ArrayList<Integer> examplesIndicies = new ArrayList<>(numExamples);

        for (int i = 0; i < numExamples; i++) {

            int rand = Util.rand(NeuralNet4.STACK_HEIGHT, states.size() - DEATH_BUFFER);

            GameState gs = states.get(rand);

            if (!gs.isTerminal) {
                examples.add(gs);
                examplesIndicies.add(rand);
            }
        }

        System.out.println("on " + examples.size() + " examples");
        net.train(examples, examplesIndicies);
        stepLastTrained = step;
        System.out.println("TRAINING COMPLETE");
    }

    /**
     * Gets the image for the state at the specified index, along with the previous (numStacked - 1) images stacked below.
     *
     * @param topIndex   The index in the state store of the topmost state
     * @param numStacked The total number of image states to stack. Should be at least one.
     * @return The stacked image
     */
    public static INDArray getStackedImg(int topIndex, int numStacked) {

        if (numStacked == 1) {
            return Vision4.process(states.get(topIndex).img);
        }

        INDArray[] arr = new INDArray[numStacked];

        for (int i = 0; i < numStacked; i++) {
            arr[i] = Vision4.process(states.get(topIndex - i).img);
        }

        return Nd4j.concat(1, arr); // apparently dimensions are [minibatchSize, channels, height, width]
    }

}