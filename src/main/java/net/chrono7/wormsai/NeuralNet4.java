package net.chrono7.wormsai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.*;

public class NeuralNet4 {

    public static final int STACK_HEIGHT = 2; // the total number of images inputted to the network (i.e. the number of previous states used to predict the future)
    public static final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    private static final int WIDTH = 200;
    private static final int HEIGHT = 100;
    private static final int CHANNELS = 1; // 3 for RGB, 1 for grayscale ??
    public static final NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    private MultiLayerNetwork net;
    private MultiLayerNetwork targetNet;
    private Random rng = new Random();
    private NormalizerMinMaxScaler rewardScaler = new NormalizerMinMaxScaler();

    public NeuralNet4() {
        net = buildNet();
//        net = loadNet();
        net.init();

        cloneTarget();

        net.addListeners(new ScoreIterationListener(10));

        DataSet rewardFit = new DataSet();

        INDArray rewardArr = Nd4j.vstack(Nd4j.create(Directions.numInstructions).assign(-100),
                Nd4j.create(Directions.numInstructions).assign(100));

        rewardFit.setFeatures(rewardArr);
        rewardScaler.fit(rewardFit);
    }

    public static INDArray pileSelf(INDArray arr, int times) {
        ArrayList<INDArray> pile = new ArrayList<>(times);
        for (int i = 0; i < times; i++) {
            pile.add(arr);
        }

        return Nd4j.pile(pile);
    }

    public static INDArray oneOn(int cols, int index, Number val, Number othersVal) {
        INDArray arr = Nd4j.create(cols).assign(othersVal);

        arr.put(0, index, val);

        return arr;
    }

    public void cloneTarget() {
        System.out.println("Cloning target network...");
        this.targetNet = net.clone();
        System.out.println("Cloned!");
    }

    public void scaleImg(INDArray img) {
        scaler.transform(img);
    }

    private MultiLayerNetwork buildNet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rng.nextLong())
//                .l2(0.005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .convolutionMode(ConvolutionMode.Same)
                .activation(Activation.RELU)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(0.00025))
                .list()
                .layer(new ConvolutionLayer.Builder(16, 16).stride(8, 8).nOut(32).build())
//                .layer(new LocalResponseNormalization.Builder().build())
//                .layer(new SubsamplingLayer.Builder(3, 3).stride(2, 2).build())
                .layer(new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).build())
                .layer(new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).build())
//                .layer(new SubsamplingLayer.Builder(3, 3).stride(2, 2).build())
                .layer(new DenseLayer.Builder().nOut(600).build())
//                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                .layer(new OutputLayer.Builder(new HuberLoss())
                        .nOut(Directions.numInstructions)
                        .activation(Activation.IDENTITY)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(HEIGHT, WIDTH, STACK_HEIGHT))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private MultiLayerNetwork loadNet() {
        MultiLayerNetwork net = null;

        try {
            net = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\model_itr_" + 50000 + ".bin", true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return net;
    }

    public int process(GameState state, boolean print) {
        return process(WormsAI.getStackedImg(state.stepIndex, STACK_HEIGHT), print);
    }

    public int process(INDArray arr, boolean print) {

        INDArray output = net.output(arr);

        if (print) {
            System.out.println(output);
        }

        return output.argMax(1).getInt(0);
    }

    private double predictTarget(GameState gs) {

        INDArray arr = WormsAI.getStackedImg(gs.stepIndex, STACK_HEIGHT);

        int bestAction = process(gs, false);

        INDArray vals = targetNet.output(arr, false, null,
                oneOn(Directions.numInstructions, bestAction, 1, 0));

        return vals.getDouble(bestAction);
    }

    public void train(Collection<GameState> gameStates) {

        Iterator<GameState> iterator = gameStates.iterator();

        LinkedList<INDArray> inputLst = new LinkedList<>();
        LinkedList<INDArray> outputLst = new LinkedList<>();
        LinkedList<INDArray> outputMaskLst = new LinkedList<>();

        GameState gs;
        while (iterator.hasNext()) {
            gs = iterator.next();
            inputLst.add(WormsAI.getStackedImg(gs.stepIndex, STACK_HEIGHT));
            outputLst.add(oneOn(Directions.numInstructions, gs.actionIndex, Q_val(gs), 0));
            outputMaskLst.add(oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
        }

        INDArray output = Nd4j.vstack(outputLst);

//        rewardScaler.transform(output);

        net.fit(Nd4j.vstack(inputLst), output, null, Nd4j.vstack(outputMaskLst));
    }

    private double Q_val(GameState gs) {
        GameState next = WormsAI.getNext(gs);
        return gs.reward + WormsAI.GAMMA * (next.isTerminal ? 0 : predictTarget(WormsAI.getNext(gs)));
    }

    public void save(int step) {
        try {
            ModelSerializer.writeModel(net, "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\model_itr_" + step + ".bin", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
