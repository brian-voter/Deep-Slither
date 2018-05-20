package net.chrono7.deepslither;

import net.chrono7.deepslither.state.GameState;
import net.chrono7.deepslither.state.StateStore;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @author Brian Voter
 */
public class Main {

    private static final boolean TRAINING_MODE = true;
    private static final int NET_DRIVE_AFTER_STEP = 5000;
    private static final boolean USE_HUMAN_START = false;
    private static final int REFRESH_DELAY = 50;
    private static final int EXPLORE_STEPS = 5;
    private static final int MIN_STEP_FOR_NET = 3000; // MAKE DIVISIBLE BY TRAIN_EVERY_N_STEPS, MAKE > CAPACITY > STACK_HEIGHT
    private static final int DEATH_BUFFER = 20;
    private static final int TRAIN_EVERY_N_STEPS = 5;
    private static final int TRAIN_N_EXAMPLES = 32;
    private static final int CLONE_TARGET_EVERY_N_STEPS = 500;
    private static final int PRINT_FREQUENCY = 10;
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END = 0.05;
    private static final double EPSILON_END_STEP = 10_000;
    private static final double EPSILON_SLOPE = (EPSILON_END - EPSILON_START) / EPSILON_END_STEP;
    private static final boolean SAVE_NET = true;
    private static final File SCORE_RECORD = new File("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\scores_" + System.currentTimeMillis() + ".txt");
    private static MouseListener mouseListener = new MouseListener();
    private static WebDriverExecutor webExe;
    private static int step = 0;
    private static int stepLastCloned = -1;
    private static int stepLastDeath = -1;
    private static long stepStartTime;
    private static NeuralNet net;
    private static int exploreUntilStep = -1;
    private static int exploreInstruction = -1;
    private static PrintWriter scoreWriter;
    private static StateStore stateStore = new StateStore(10000, DEATH_BUFFER);


    public static void main(String[] args) {

        net = new NeuralNet();

        registerMouseListener();

        webExe = new WebDriverExecutor();
        webExe.navigate();

        try {
            scoreWriter = new PrintWriter(SCORE_RECORD);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

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


    private static void engage() throws InterruptedException {

        GameState prev = null;

        while (true) {

            stepStartTime = System.currentTimeMillis();

            if (step % PRINT_FREQUENCY == 0) {
                System.out.println("step: " + step);
            }

            if (webExe.testLoss()) {
                System.out.println("LOST");

                scoreWriter.println(Objects.requireNonNull(prev).score);
                scoreWriter.flush();

                stateStore.notifyDeath();

                stepLastDeath = step;


                webExe.fixLoss();
                Thread.sleep(2000);
            }

            if (step % 10000 == 0 && SAVE_NET) {
                net.save(step);
            }

            INDArray imgBefore = prev != null ? prev.after : Vision.frame2INDArray(webExe.getScreenshot());
            GameState gs = new GameState(imgBefore);

            gs.actionIndex = drive(gs.before);

            if (step >= MIN_STEP_FOR_NET && step % TRAIN_EVERY_N_STEPS == 0) {
                if (step - stepLastCloned > CLONE_TARGET_EVERY_N_STEPS || step == MIN_STEP_FOR_NET) {
                    net.cloneTarget();
                    stepLastCloned = step;
                }
                train();
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

            gs.after = Vision.frame2INDArray(webExe.getScreenshot());
            stateStore.push(gs);

            prev = gs;

            step++;

        }
    }

    private static void delay() throws InterruptedException {
        long desiredStopTime = stepStartTime + REFRESH_DELAY;

        long cur = System.currentTimeMillis();

        if (desiredStopTime > cur) {
            Thread.sleep(desiredStopTime - cur);
        }
    }

    /**
     * Moves the mouse and boosts based on the action policy.
     *
     * @param image the input state
     * @return the action taken
     */
    private static int drive(INDArray image) {
        int actionIndex;
        boolean do_move = true;

        if (TRAINING_MODE) {

            if (step < NET_DRIVE_AFTER_STEP) {

                if (USE_HUMAN_START) {
                    //TODO: fix human start?
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
                    double epsilon = Math.max(EPSILON_SLOPE * step + EPSILON_START, EPSILON_END);

                    if (Math.random() < epsilon) { // If starting a random period
                        actionIndex = getRandomAction();
                    } else { // AI selects
                        actionIndex = net.predictBestAction(image, step % PRINT_FREQUENCY == 0);
                    }
                }
            }
        } else {
            actionIndex = (step <= NeuralNet.STACK_HEIGHT ? getRandomAction() :
                    net.predictBestAction(image, step % PRINT_FREQUENCY == 0));
        }

        if (do_move) {
            GameInstruction action = Directions.getInstruction(actionIndex);
            webExe.act(action);
        }

        return actionIndex;
    }

    private static int getRandomAction() {

        if (step > exploreUntilStep) {
            exploreUntilStep = step + EXPLORE_STEPS;
            exploreInstruction = Directions.randomIndex();
        }

        return exploreInstruction;
    }

    private static void train() {
        ArrayList<Pair<Integer, GameState>> sample = stateStore.sample(TRAIN_N_EXAMPLES);
        System.out.println("training on " + sample.size() + " examples");
        INDArray error = net.train(sample);

        for (int i = 0; i < error.rows(); i++) {
            stateStore.update(sample.get(i).getFirst(), error.getInt(i, 0));
        }

        System.out.println("TRAINING COMPLETE");
    }

}