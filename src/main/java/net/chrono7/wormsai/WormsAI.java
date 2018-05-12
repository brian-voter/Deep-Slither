package net.chrono7.wormsai;

import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class WormsAI {

    private static final int REFRESH_DELAY = 100;
    private static final double EXPLORE_ODDS = 0.005; // epsilon
    private static final int EXPLORE_STEPS = 20;
    private static final int MIN_STEP_FOR_NET = 50;
    private static final int DEATH_BUFFER = 15;
    private static final int TRAIN_EVERY_N_STEPS = 3;
    private static ArrayList<GameState> states = new ArrayList<>();
    private static WebDriverExecutor webExe;
    private static int step = 0;
    private static int stepOfLastDeath = -1;
    private static MouseListener mouseListener = new MouseListener();
    private static NeuralNet4 net;
    private static Random rng = new Random();
    private static int exploreUntilStep = -1;
    private static int exploreInstruction = -1;

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        net = new NeuralNet4();

        registerMouseListener();

        webExe = new WebDriverExecutor();
        webExe.navigate();

        captureLoop();
        engageAI();
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

    private static void delayStep(long stepStartTime) throws InterruptedException {
        long stepFinishTime;
        stepFinishTime = System.currentTimeMillis();

        if (stepStartTime + REFRESH_DELAY > stepFinishTime) {
            Thread.sleep(stepStartTime + REFRESH_DELAY - stepFinishTime);
        }
    }

    private static void captureLoop() {
        try {
            while (step < 500) {

                BufferedImage capture = webExe.getScreenshot();

                if (webExe.testLoss()) { // Do training if we lost
                    System.out.println("LOSS");

                    for (int i = states.size() - 1; i > states.size() - 10; i--) {
                        states.get(i).reward = -100;
                    }

                    ImageIO.write(capture, "png",
                            new File("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\out.png"));

                    stepOfLastDeath = step;
                    webExe.fixLoss();
                    Thread.sleep(1000);
                }

                INDArray img = NeuralNet4.loader.asMatrix(capture);
                net.scaleImg(img);

                GameState gs = new GameState(img, step);
                states.add(gs);

                int actionIndex = Directions.getClosest(MouseInfo.getPointerInfo().getLocation(),
                        mouseListener.isMousePressed());

                GameInstruction action = Directions.getInstruction(actionIndex);

                int scorePre = webExe.getScore();

                System.out.println(action.point + " " + action.boost);

//                delayStep(stepStartTime); // "do the action"
                Thread.sleep(REFRESH_DELAY);

                int scorePost = webExe.getScore();

                gs.augment(actionIndex, scorePost, scorePost - scorePre);

                if (step > MIN_STEP_FOR_NET && step % TRAIN_EVERY_N_STEPS == 0) {
//                    train();
                    train2();
                }

                step++;

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        engageAI();
    }

    private static void engageAI() {

        System.out.println("AUTO PILOT ENGAGE!!!\n\n");

        long stepStartTime;

        try {
            while (true) {

                stepStartTime = System.currentTimeMillis();

                BufferedImage capture = webExe.getScreenshot();

                if (webExe.testLoss()) { // Do training if we lost
                    System.out.println("LOSS");

                    for (int i = states.size() - 1; i > states.size() - 5; i--) {
                        states.get(i).reward = 0;
                    }

                    for (int i = states.size() - 5; i > states.size() - DEATH_BUFFER; i--) {
                        states.get(i).reward = -100;
                    }

                    ImageIO.write(capture, "png",
                            new File("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\out.png"));

//                    if (step > MIN_STEP_FOR_NET && step - stepOfLastDeath > 20) {
//
//                        //TODO: DO MASKING on outputs not trained
//                        train();
//                    }

                    stepOfLastDeath = step;
                    webExe.fixLoss();
                    Thread.sleep(1000);
                }

                INDArray img = NeuralNet4.loader.asMatrix(capture);
                net.scaleImg(img);

                GameState gs = new GameState(img, step);
                states.add(gs);

                int actionIndex = selectAction(gs);

                GameInstruction action = Directions.getInstruction(actionIndex);

                int scorePre = webExe.getScore();

                System.out.println(action + (step < exploreUntilStep ? " exploring" : " net"));

                webExe.pointAdjusted(action.point);
                webExe.setBoost(action.boost);

//                delayStep(stepStartTime); // "do the action"
                Thread.sleep(REFRESH_DELAY);

                int scorePost = webExe.getScore();

                gs.augment(actionIndex, scorePost, scorePost - scorePre);

                if (step > MIN_STEP_FOR_NET && step % TRAIN_EVERY_N_STEPS == 0) {
//                    train();
                    train2();
                }

                step++;

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void train2() {
        int numExamples = 20;

        ArrayList<GameState> examples = new ArrayList<>(numExamples);

        for (int i = 0; i < numExamples; i++) {
            examples.add(states.get(
                    rng.nextInt(states.size() - DEATH_BUFFER) + DEATH_BUFFER - 1));
        }

        System.out.println("on " + examples.size() + " examples");
        net.train(examples);
        System.out.println("TRAINING COMPLETE");
    }

    private static void train() {
        System.out.println("TRAINING");
        int numExamples = Math.min((step - stepOfLastDeath) * 4, 1000);

        ArrayList<GameState> examples = new ArrayList<>(numExamples);

        for (int i = 0; i < numExamples; i++) {
            examples.add(states.get(
                    rng.nextInt(states.size() - NeuralNet4.STACK_HEIGHT) + NeuralNet4.STACK_HEIGHT - 1));
        }

        System.out.println("on " + examples.size() + " examples");
        net.train(examples);
        System.out.println("TRAINING COMPLETE");
    }

    private static int selectAction(GameState state) {
        if (step <= exploreUntilStep) { // Keep exploring
            return exploreInstruction;
        } else {
            if (rng.nextDouble() < EXPLORE_ODDS || step < MIN_STEP_FOR_NET) { // Start exploring
                exploreUntilStep = step + EXPLORE_STEPS;
                exploreInstruction = Directions.randomIndex();
                return exploreInstruction;

            } else { // Use network
                return net.process(state);
            }
        }
    }

    /**
     * Gets the image for the state at the specified step, along with the previous numStacked images stacked below.
     *
     * @param step       The step number for the topmost image state
     * @param numStacked The total number of image states to stack. Should be at least one.
     * @return The stacked image
     */
    public static INDArray getStackedImg(int step, int numStacked) {
        INDArray[] arr = new INDArray[numStacked];

        for (int i = 0; i < numStacked; i++) {
            arr[i] = (states.get(step - i).img);
        }

        return Nd4j.concat(1, arr);
    }

}