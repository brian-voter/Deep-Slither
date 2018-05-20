package net.chrono7.deepslither;

import net.chrono7.deepslither.state.GameState;
import org.datavec.image.loader.Java2DNativeImageLoader;
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
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Random;

/**
 * @author Brian Voter
 */
public class NeuralNet {

    public static final int STACK_HEIGHT = 1; // the total number of images inputted to the network (i.e. the number of previous states used to predict the future)
    public static final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    public static final int WIDTH = 200;
    public static final int HEIGHT = 100;
    public static final int CHANNELS = 1;
    public static final Java2DNativeImageLoader loader = new Java2DNativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    private static final double GAMMA = 0.99;
    private MultiLayerNetwork onlineNet;
    private MultiLayerNetwork targetNet;
    private Random rng = new Random();

    public NeuralNet() {
        onlineNet = buildNet();
        onlineNet.init();

        cloneTarget();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage memoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(memoryStatsStorage);

        StatsStorage fileStatsStorage = new FileStatsStorage(
                new File("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\stats_" +
                        System.currentTimeMillis() + ".dl4j"));

        onlineNet.addListeners(new StatsListener(fileStatsStorage), new StatsListener(memoryStatsStorage),
                new ScoreIterationListener(10));

    }

    public static INDArray oneOn(int cols, int index, Number val, Number othersVal) {
        INDArray arr = Nd4j.create(cols).assign(othersVal);

        arr.put(0, index, val);

        return arr;
    }

    public void cloneTarget() {
        System.out.println("Cloning target network...");
        this.targetNet = onlineNet.clone();
        System.out.println("Cloned!");
    }

    private MultiLayerNetwork buildNet() {

        //Dueling graph
//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rng.nextLong())
////                .l2(0.005)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .activation(Activation.RELU)
//                .weightInit(WeightInit.RELU)
//                .updater(new Adam(0.0001))
//                .graphBuilder()
//                .addInputs("input")
//                .addLayer("c1", new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(16).build(), "input")
//                .addLayer("c2", new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(32).gradientNormalization(GradientNormalization.ClipL2PerLayer).gradientNormalizationThreshold(10).build(), "c1")
////                .addLayer("c3", new ConvolutionLayer.Builder(3, 3).gradientNormalization(GradientNormalization.ClipL2PerLayer).gradientNormalizationThreshold(10).padding(2, 2).stride(1, 1).nOut(32).build(), "c2")
//                .addVertex("gradScaler", new GradientScalerVertex(), "c2")
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
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(0.0005))
                .list()
                .layer(new ConvolutionLayer.Builder(8, 8).stride(4, 4).nOut(16).build())
                .layer(new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(32).build())
                .layer(new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).build())
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

    private MultiLayerNetwork loadNet(String path) {
        MultiLayerNetwork net = null;

        try {
            net = ModelSerializer.restoreMultiLayerNetwork(path, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return net;
    }

    public int predictBestAction(INDArray arr, boolean print) {
        INDArray input = Nd4j.zeros(1, STACK_HEIGHT, HEIGHT, WIDTH);
        input.putRow(0, arr);

//        INDArray output = onlineNet.outputSingle(false, input);
        INDArray output = onlineNet.output(input, false);

        if (print) {
            System.out.println(output);
        }

        return output.argMax(1).getInt(0);
    }

    /**
     * Returns the highest quality action for the input states
     * according to the online network.
     *
     * @param inputs The input states
     * @return A column vector containing the best action for each input state
     */
    private INDArray predictBestAction(INDArray inputs) {

//        INDArray output = onlineNet.outputSingle(false, inputs);
        INDArray output = onlineNet.output(inputs, false);

        return output.argMax(1); // returns [inputs.rows x 1] the best action index
    }

    /**
     * Returns the Q values of the input states, where the action argument is the best action as
     * decided by the online network and the Q value is determined by the target network
     *
     * @param inputs The states to obtain the Q values for
     * @return The Q values as a column vector
     */
    private INDArray predictQValue(INDArray inputs) {
        return predictQValue(inputs, predictBestAction(inputs), targetNet);
    }

    /**
     * Returns the Q values of the inputs according to useNet, where the action index is provided by the column vector actions.
     * Q(s, a)
     *
     * @param inputs  The states as arguments to the Q function
     * @param actions The actions as arguments to the Q function
     * @param useNet  The network used to make the predictions
     * @return The Q values as a column vector
     */
    private INDArray predictQValue(INDArray inputs, INDArray actions, MultiLayerNetwork useNet) {
//    private INDArray predictQValue(INDArray inputs, INDArray actions, ComputationGraph useNet) {

//        INDArray output = useNet.outputSingle(false, inputs); //[inputs.rows x nActions] the Q vals for each action
        INDArray output = useNet.output(inputs, false); //[inputs.rows x nActions] the Q vals for each action

        INDArray ret = Nd4j.zeros(output.rows(), 1);

        for (int row = 0; row < ret.rows(); row++) {
            ret.put(row, 0, output.getScalar(row, actions.getInt(row, 0)));
        }

        return ret;
    }

    /**
     * Trains the network(s) using the states
     *
     * @param states the states to train on
     * @return a column vector containing the error for each state
     */
    public INDArray train(Collection<Pair<Integer, GameState>> states) {
        INDArray inputs = Nd4j.zeros(states.size(), STACK_HEIGHT, HEIGHT, WIDTH);
        INDArray actionsTaken = Nd4j.zeros(states.size(), 1);
        INDArray nextInputs = Nd4j.zeros(states.size(), STACK_HEIGHT, HEIGHT, WIDTH);
        INDArray labelMask = Nd4j.zeros(states.size(), Directions.numInstructions);
        INDArray rewards = Nd4j.zeros(states.size(), 1);
        INDArray nextNotTerminal = Nd4j.zeros(states.size(), 1);

        int row = 0;
        for (Pair<Integer, GameState> elem : states) {
            GameState gs = elem.getRight();
            inputs.putRow(row, gs.before);
            nextInputs.putRow(row, gs.after);
            actionsTaken.put(row, 0, gs.actionIndex);
            labelMask.putRow(row, oneOn(Directions.numInstructions, gs.actionIndex, 1, 0));
            rewards.put(row, 0, gs.reward);
            nextNotTerminal.put(row, 0, !gs.isTerminal ? 1 : 0);
            row++;
        }

        INDArray labels = Q_Val(rewards, nextInputs, nextNotTerminal, labelMask); //labels = target
        INDArray predictions = predictQValue(inputs, actionsTaken, onlineNet);

//        onlineNet.fit(new INDArray[]{inputs}, new INDArray[]{labels}, null, new INDArray[]{labelMask});
        onlineNet.fit(inputs, labels, null, labelMask);

        //TODO: Fix bug when labels.rows() = 1
        return Transforms.abs(labels.max(0).sub(predictions));
    }

    /**
     * Gets the target Q values for the specified argument inputs
     *
     * @param rewards         The rewards for each transition as a column vector
     * @param nextInputs      The images for the post transition states
     * @param nextNotTerminal A column vector s.t. each row contains 1 if the post state is not terminal, 0 otherwise
     * @param actions         The action causing the transition: a matrix where each row contains 0s except for one
     *                        1 in the column whose index matches the action taken. Aka the label mask
     * @return The Q value labels for each example. Shape is [nExamples, nActions]. For each row, all entries are 0
     * except for the entry in the column of the action taken
     */
    private INDArray Q_Val(INDArray rewards, INDArray nextInputs, INDArray nextNotTerminal, INDArray actions) {
        INDArray gammaQ = predictQValue(nextInputs).muli(GAMMA).muli(nextNotTerminal);
        return actions.muli(rewards.add(gammaQ));
    }

    public void save(int step) {
        try {
            ModelSerializer.writeModel(onlineNet, "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\misc\\model_itr_" + step + ".bin", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
