package net.chrono7.wormsai;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.*;

public class NeuralNet4 {

    public static final int STACK_HEIGHT = 1; // the total number of images inputted to the network (i.e. the number of previous states used to predict the future)
    public static final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    public static final int WIDTH = 200;
    public static final int HEIGHT = 100;
    public static final int CHANNELS = 1; // 3 for RGB, 1 for grayscale ??
    public static final Java2DNativeImageLoader loader = new Java2DNativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    private ComputationGraph net;
    private ComputationGraph targetNet;
    private Random rng = new Random();
//    private NormalizerMinMaxScaler rewardScaler = new NormalizerMinMaxScaler();

    public NeuralNet4() {
        net = buildNet();
//        net = loadNet();
        net.init();

        cloneTarget();

        net.addListeners(new ScoreIterationListener(10));

//        DataSet rewardFit = new DataSet();

//        INDArray rewardArr = Nd4j.vstack(Nd4j.create(Directions.numInstructions).assign(-10),
//                Nd4j.create(Directions.numInstructions).assign(150));

//        rewardFit.setFeatures(rewardArr);
//        rewardScaler.fit(rewardFit);
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

    private ComputationGraph buildNet() {

        ComputationGraphConfiguration confG = new NeuralNetConfiguration.Builder()
                .seed(rng.nextLong())
//                .l2(0.005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
//                .updater(new RmsProp(0.00025, 0.95, 0.01))
                .updater(new Adam(0.00025))
                .graphBuilder()
                .addInputs("input")
                .addLayer("c1", new ConvolutionLayer.Builder(16, 16).stride(8, 8).nOut(32).build(), "input")
                .addLayer("c2", new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(64).build(), "c1")
                .addLayer("c3", new ConvolutionLayer.Builder(3, 3).padding(2, 2).stride(1, 1).nOut(64).build(), "c2")

                // SPLIT - Advantage
                .addLayer("advantageDense", new DenseLayer.Builder().nOut(512).build(), "c3")
                .addLayer("advantageOut", new DenseLayer.Builder().nOut(Directions.numInstructions).build(), "advantageDense")
                .addVertex("advantageAverage", new AverageVertex(), "advantageOut")
                .addVertex("advantageRepeat", new RepeatVertex(Directions.numInstructions), "advantageAverage")

                .addVertex("advantageSubtr", new ElementWiseVertex(ElementWiseVertex.Op.Subtract), "advantageRepeat", "advantageOut")

                // SPLIT - Value

                .addLayer("valueDense", new DenseLayer.Builder().nOut(512).build(), "c3")
                .addLayer("valueOut", new DenseLayer.Builder().nOut(1).build(), "valueDense")
                .addVertex("valueRepeat", new RepeatVertex(Directions.numInstructions), "valueOut")


                .addVertex("add", new ElementWiseVertex(ElementWiseVertex.Op.Add),"advantageSubtr", "valueRepeat")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                        .addLayer("output", new OutputLayer.Builder(new LogCoshLoss())
                        .nOut(Directions.numInstructions)
                        .activation(Activation.IDENTITY)
                        .build(), "add")
                .setOutputs("output")
                .setInputTypes(InputType.convolutional(HEIGHT, WIDTH, STACK_HEIGHT))
                .backprop(true).pretrain(false)
                .build();
//                .add


//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rng.nextLong())
////                .l2(0.005)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .activation(Activation.RELU)
//                .weightInit(WeightInit.XAVIER)
////                .updater(new RmsProp(0.00025, 0.95, 0.01))
//                .updater(new Adam(0.0001))
//                .list()
//                .layer(new ConvolutionLayer.Builder(16, 16).stride(8, 8).nOut(32).build())
//                .layer(new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(64).build())
//                .layer(new ConvolutionLayer.Builder(3, 3).padding(2, 2).stride(1, 1).nOut(64).build())
//                .layer(new DenseLayer.Builder().nOut(512).build())
//                .layer(new OutputLayer.Builder(new LogCoshLoss())
//                        .nOut(Directions.numInstructions)
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .backprop(true).pretrain(false)
//                .setInputType(InputType.convolutional(HEIGHT, WIDTH, STACK_HEIGHT))
//                .build();

        return new ComputationGraph(confG);
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

    /**
     * Returns the index of the action predicted to be "best" given the state arr.
     *
     * @param state The game state
     * @param print Set to true to print the predicted Q values for each action
     * @return the index of the best action
     */
    public int predictBestAction(GameState state, boolean print) {
        return predictBestAction(WormsAI.getStackedImg(state, STACK_HEIGHT), print);
    }

    private int predictBestAction(INDArray arr, boolean print) {

        INDArray output = net.outputSingle(false, arr);

        if (print) {
            System.out.println(output);
        }

        return output.argMax(1).getInt(0);
    }

    private INDArray predictBestAction(INDArray inputs) {

        INDArray output = net.outputSingle(false, inputs);

        return output.argMax(1);
    }

    private double predictQValue(GameState gs) {

        INDArray arr = WormsAI.getStackedImg(gs, STACK_HEIGHT);

        int bestAction = predictBestAction(arr, false);

        INDArray output = targetNet.outputSingle(false, arr);

        return output.getDouble(bestAction);
    }

    private INDArray predictQValue(INDArray inputs) {
        INDArray bestActions = predictBestAction(inputs); // [inputs.rows x 1] the best action index

        INDArray output = targetNet.outputSingle(false, inputs); //[inputs.rows x nActions] the Q vals for each action

        INDArray ret = Nd4j.zeros(inputs.rows(), 1);

        for (int row = 0; row < ret.rows(); row++) {
            ret.put(row, 0, output.getScalar(row, bestActions.getInt(row, 0)));
        }

        return ret;
    }

    public void train(Collection<GameState> gameStates) {

        Iterator<GameState> iterator = gameStates.iterator();

        LinkedList<INDArray> inputLst = new LinkedList<>();
        LinkedList<INDArray> labelLst = new LinkedList<>();
        LinkedList<INDArray> labelMaskLst = new LinkedList<>();

        INDArray inputs = Nd4j.zeros(gameStates.size(), Directions.numInstructions);
        INDArray nextInputs = Nd4j.zeros(gameStates.size(), Directions.numInstructions);
        INDArray labelMask = Nd4j.zeros(gameStates.size(), Directions.numInstructions);
        INDArray nextRewards = Nd4j.zeros(gameStates.size(), 1);
        INDArray nextNotTerminal = Nd4j.zeros(gameStates.size(), 1);

        GameState gs;
        int row = 0;
        while (iterator.hasNext()) {
            gs = iterator.next();
            GameState ns = WormsAI.getNext(gs);
            inputs.putRow(row, WormsAI.getStackedImg(gs, STACK_HEIGHT));
            nextInputs.putRow(row, WormsAI.getStackedImg(ns, STACK_HEIGHT));
            labelMask.putRow(row, oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
            nextRewards.put(row, 0, ns.reward);
            nextNotTerminal.put(row, 0, !ns.isTerminal ? 1 : 0);
        }

        INDArray labels = Q_Val(nextRewards, nextInputs, nextNotTerminal);

//        while (iterator.hasNext()) {
//            gs = iterator.next();
//            inputLst.add(WormsAI.getStackedImg(gs, STACK_HEIGHT));
//            labelLst.add(oneOn(Directions.numInstructions, gs.actionIndex, Q_val(gs), 0));
//            labelMaskLst.add(oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
//        }

//        INDArray labels = Nd4j.vstack(labelLst);

//        rewardScaler.transform(labels);

        net.fit(new INDArray[]{Nd4j.vstack(inputLst)}, new INDArray[]{labels}, null, new INDArray[]{Nd4j.vstack(labelMaskLst)});
    }

    private INDArray Q_Val(INDArray nextRewards, INDArray nextInputs, INDArray nextNotTerminal) {
        INDArray gammaQ = predictQValue(nextInputs).mul(WormsAI.GAMMA).mul(nextNotTerminal);
        return nextRewards.add(gammaQ);
    }

    private double Q_val(GameState gs) {
        GameState next = WormsAI.getNext(gs);
        return next.reward + WormsAI.GAMMA * (next.isTerminal ? 0 : predictQValue(next));
    }

    public void save(int step) {
        try {
            ModelSerializer.writeModel(net, "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\model_itr_" + step + ".bin", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
