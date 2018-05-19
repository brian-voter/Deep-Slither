package net.chrono7.wormsai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Random;

public class NeuralNet4 {

    public static final int STACK_HEIGHT = 1; // the total number of images inputted to the network (i.e. the number of previous states used to predict the future)
    public static final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    public static final int WIDTH = 200;
    public static final int HEIGHT = 100;
    public static final int CHANNELS = 1; // 3 for RGB, 1 for grayscale ??
    public static final NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    private MultiLayerNetwork net;
    private MultiLayerNetwork targetNet;
    private Random rng = new Random();
//    private NormalizerMinMaxScaler rewardScaler = new NormalizerMinMaxScaler();

    public NeuralNet4() {

        net = buildNet();
//        net = loadNet();
        net.init();

        cloneTarget();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage memoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(memoryStatsStorage);

        StatsStorage fileStatsStorage = new FileStatsStorage(
                new File("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\stats_" +
                        System.currentTimeMillis() + ".dl4j"));

        net.addListeners(new StatsListener(fileStatsStorage), new StatsListener(memoryStatsStorage),
                new ScoreIterationListener(10));

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

    private MultiLayerNetwork buildNet() {

//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rng.nextLong())
////                .l2(0.005)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .activation(Activation.RELU)
//                .weightInit(WeightInit.XAVIER)
////                .weightInit(WeightInit.DISTRIBUTION)
////                .dist(new NormalDistribution(0.0025, 0.01))
////                .updater(new RmsProp(0.00025, 0.95, 0.01))
//                .updater(new Adam(0.00015)) //slightly lower
//                .graphBuilder()
//                .addInputs("input")
//                .addLayer("c1", new ConvolutionLayer.Builder(16, 16).stride(8, 8).nOut(16).build(), "input")
//                .addLayer("c2", new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(32).build(), "c1")
//                .addLayer("c3", new ConvolutionLayer.Builder(3, 3).gradientNormalization(GradientNormalization.ClipL2PerLayer).gradientNormalizationThreshold(10).padding(2, 2).stride(1, 1).nOut(32).build(), "c2")
//                .addVertex("gradScaler", new GradientScalerVertex(), "c3")
//
//                // SPLIT - Advantage
//                .addLayer("advantageDense", new DenseLayer.Builder().nOut(256).build(), "gradScaler")
//                .addLayer("advantageOut", new DenseLayer.Builder().nOut(Directions.numInstructions).build(), "advantageDense")
//                .addVertex("advantageAverage", new AverageVertex(), "advantageOut")
//                .addVertex("advantageAvgRepeat", new RepeatVertex(Directions.numInstructions), "advantageAverage")
//
//                .addVertex("advantageSubtr", new ElementWiseVertex(ElementWiseVertex.Op.Subtract), "advantageOut", "advantageAvgRepeat")
//
//                // SPLIT - Value
//
//                .addLayer("valueDense", new DenseLayer.Builder().nOut(256).build(), "gradScaler")
//                .addLayer("valueOut", new DenseLayer.Builder().nOut(1).build(), "valueDense")
//                .addVertex("valueRepeat", new RepeatVertex(Directions.numInstructions), "valueOut")
//
//
//                .addVertex("add", new ElementWiseVertex(ElementWiseVertex.Op.Add), "advantageSubtr", "valueRepeat")
////                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                .addLayer("output", new OutputLayer.Builder(new LogCoshLoss())
//                        .nOut(Directions.numInstructions)
//                        .activation(Activation.IDENTITY)
//                        .build(), "add")
//                .setOutputs("output")
//                .setInputTypes(InputType.convolutional(HEIGHT, WIDTH, STACK_HEIGHT))
//                .backprop(true).pretrain(false)
//                .build();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rng.nextLong())
//                .l2(0.005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
//                .weightInit(WeightInit.DISTRIBUTION)
//                .dist(new NormalDistribution(0.0025, 0.01))
//                .updater(new RmsProp(0.00025, 0.95, 0.01))
                .updater(new Adam(0.00025))
//                .updater(new RmsProp(0.00025, 0.95, 0.01))
                .list()
                .layer(new ConvolutionLayer.Builder(16, 16).stride(8, 8).nOut(32).build())
                .layer(new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(64).build())
                .layer(new ConvolutionLayer.Builder(3, 3).padding(2, 2).stride(1, 1).nOut(64).build())
                .layer(new DenseLayer.Builder().nOut(256).build())
//                .layer(new OutputLayer.Builder(new LogCoshLoss())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
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

    /**
     * Returns the index of the action predicted to be "best" given the state arr.
     *
     * @param stateIndex The index of the state in the state store
     * @param print      Set to true to print the predicted Q values for each action
     * @return the index of the best action
     */
    public int predictBestAction(int stateIndex, boolean print) {
        INDArray input = Nd4j.zeros(1, STACK_HEIGHT, HEIGHT, WIDTH);
        input.putRow(0, WormsAI.getStackedImg(stateIndex, STACK_HEIGHT));
        return predictBestAction(input, print);
    }

    private int predictBestAction(INDArray arr, boolean print) {

//        INDArray output = net.outputSingle(false, arr);
        INDArray output = net.output(arr, false);

        if (print) {
            System.out.println(output);
        }

        return output.argMax(1).getInt(0);
    }

    //FOR MATRIX
    private INDArray predictBestAction(INDArray arr) {

//        INDArray output = net.outputSingle(false, arr);
        INDArray output = net.output(arr, false);

        return output.argMax(1);
    }

//    private double predictQValue(int stateIndex) {
//
//        INDArray arr = WormsAI.getStackedImg(stateIndex, STACK_HEIGHT);
//
//        int bestAction = predictBestAction(arr, false);
//
//        INDArray output = targetNet.outputSingle(false, arr);
////        INDArray output = targetNet.output(arr, false);
//
//        return output.getDouble(bestAction);
//    }

    //FOR MATRIX
    private INDArray predictQValue(INDArray arr) {
        INDArray bestActions = predictBestAction(arr); // [inputs.rows x 1] the best action index

//        INDArray output = targetNet.outputSingle(false, arr); //[inputs.rows x nActions] the Q vals for each action
        INDArray output = targetNet.output(arr, false); //[inputs.rows x nActions] the Q vals for each action

        INDArray ret = Nd4j.zeros(output.rows(), 1);

        for (int row = 0; row < ret.rows(); row++) {
            ret.put(row, 0, output.getScalar(row, bestActions.getInt(row, 0)));
        }

        return ret;
    }

    public void train(Collection<GameState> gameStates, ArrayList<Integer> examplesIndicies) {

        Iterator<GameState> iterator = gameStates.iterator();
//
//        LinkedList<INDArray> inputLst = new LinkedList<>();
//        LinkedList<INDArray> labelLst = new LinkedList<>();
//        LinkedList<INDArray> labelMaskLst = new LinkedList<>();

        INDArray inputs = Nd4j.zeros(gameStates.size(), STACK_HEIGHT, HEIGHT, WIDTH);
        INDArray nextInputs = Nd4j.zeros(gameStates.size(), STACK_HEIGHT, HEIGHT, WIDTH);
        INDArray labelMask = Nd4j.zeros(gameStates.size(), Directions.numInstructions);
        INDArray rewards = Nd4j.zeros(gameStates.size(), 1);
        INDArray nextNotTerminal = Nd4j.zeros(gameStates.size(), 1);

        GameState gs;
        int row = 0;
        while (iterator.hasNext()) {
            gs = iterator.next();
            GameState ns = WormsAI.getState(examplesIndicies.get(row) + 1);
            inputs.putRow(row, WormsAI.getStackedImg(examplesIndicies.get(row), STACK_HEIGHT));
            nextInputs.putRow(row, WormsAI.getStackedImg(examplesIndicies.get(row), STACK_HEIGHT));
            labelMask.putRow(row, oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
            rewards.put(row, 0, gs.reward);
            nextNotTerminal.put(row, 0, !ns.isTerminal ? 1 : 0);
            row++;
        }


        INDArray labels = Q_Val(rewards, nextInputs, nextNotTerminal, labelMask);
//        for (int i = 0; i < labels.rows(); i++) {
//            System.out.println(labels.getRow(i));
//        }

//        while (iterator.hasNext()) {
//            gs = iterator.next();
//            inputLst.add(WormsAI.getStackedImg(gs, STACK_HEIGHT));
//            labelLst.add(oneOn(Directions.numInstructions, gs.actionIndex, Q_val(gs), 0));
//            labelMaskLst.add(oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
//        }

//        INDArray labels = Nd4j.vstack(labelLst);

//        rewardScaler.transform(labels);

//        net.fit(new INDArray[]{inputs}, new INDArray[]{labels}, null, new INDArray[]{labelMask});
        net.fit(inputs, labels, null, labelMask);
    }

    private INDArray Q_Val(INDArray rewards, INDArray nextInputs, INDArray nextNotTerminal, INDArray actions) {
        INDArray gammaQ = predictQValue(nextInputs).muli(WormsAI.GAMMA).muli(nextNotTerminal);
        return actions.muli(rewards.add(gammaQ));
    }

//    private double Q_val(GameState gs) {
//        GameState next = WormsAI.getNext(gs);
//        return next.reward + WormsAI.GAMMA * (next.isTerminal ? 0 : predictQValue(next));
//    }

    public void save(int step) {
        try {
            ModelSerializer.writeModel(net, "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\model_itr_" + step + ".bin", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
