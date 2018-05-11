//package net.chrono7.wormsai;
//
//import com.google.common.annotations.VisibleForTesting;
//import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
//import org.datavec.api.split.CollectionInputSplit;
//import org.datavec.api.split.FileSplit;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.api.storage.StatsStorage;
//import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
//import org.deeplearning4j.nn.conf.ConvolutionMode;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.graph.MergeVertex;
//import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.stats.StatsListener;
//import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerHybrid;
//import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.learning.config.Nesterovs;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.nd4j.linalg.schedule.ScheduleType;
//import org.nd4j.linalg.schedule.StepSchedule;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.File;
//import java.io.IOException;
//import java.net.URI;
//import java.util.*;
//import java.util.stream.Collectors;
//
///**
// * Animal Classification
// * <p>
// * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
// * <p>
// * References:
// * - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
// * - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
// * <p>
// * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
// * - Add additional images to the dataset
// * - Apply more transforms to dataset
// * - Increase epochs
// * - Try different model configurations
// * - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
// */
//
//public class NeuralNet3 {
//
//    protected static final Logger log = LoggerFactory.getLogger(NeuralNet3.class);
//    protected static int batchSize = 50;
//    protected static int epochs = 20;
//    protected static double splitTrainTest = 0.8;
//    protected static boolean save = true;
//    protected static String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out
//    private static int height = 200;
//    private static int width = 200;
//    private static int channels = 3;
//    //    protected static long seed = 42;
//    private static Random rng = new Random();
//    //    private final INDArray pointsArr;
//    private ImagePreProcessingScaler imgScaler;
//    private NormalizerMinMaxScaler pointScaler;
//    private NormalizerMinMaxScaler scoreScaler;
//    private int outputNum = 1;
//    private ComputationGraph net;
//    private NativeImageLoader loader = new NativeImageLoader(150, 150, 3);
//    private List<String> labelNames = Arrays.asList("FOOD", "PREY", "WORM");
//
////    int[] xVals = new int[]{50, 450, 850, 1150, 1350, 1550};
////    int[] yVals = new int[]{50, 170, 290, 410, 530, 650, 770, 890};
//
//    public NeuralNet3() throws IOException {
//
//        net = customNet();
//        net.init();
//
//        System.out.println("Loading capture names...");
//
//        File imgDir = new File(WormsAI2.STATES_CAPTURE_DIR);
//        List<URI> uris = Arrays.stream(Objects.requireNonNull(imgDir.listFiles()))
//                .map(File::toURI).sorted().collect(Collectors.toList());
//
//        CollectionInputSplit cis = new CollectionInputSplit(uris);
//
//        ImageRecordReader imageReader = new ImageRecordReader(height, width, channels);
//        imageReader.initialize(cis);
//
//        System.out.println("Preparing data import...");
//
//        CSVRecordReader featureCSV = new CSVRecordReader();
//        try {
//            featureCSV.initialize(new FileSplit(new File(WormsAI2.STATES_FEATURES_FILE)));
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//        CSVRecordReader labelCSV = new CSVRecordReader();
//        try {
//            labelCSV.initialize(new FileSplit(new File(WormsAI2.STATES_LABELS_FILE)));
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//        RecordReaderMultiDataSetIterator msi = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//                .addReader("imageReader", imageReader).addInput("imageReader")
//                .addReader("featureCSV", featureCSV).addInput("featureCSV")
//                .addReader("labelCSV", labelCSV).addOutput("labelCSV")
//                .build();
//
//        MultiNormalizerHybrid norm = new MultiNormalizerHybrid().minMaxScaleAllInputs().minMaxScaleAllOutputs();
//        norm.fit(msi);
//        msi.setPreProcessor(norm);
//
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
//
//        System.out.println("Training...");
//
//        for (int i = 0; i < epochs; i++) {
//            net.fit(msi);
//            System.out.println("Completed Epoch " + i);
//        }
//
//        System.out.println("Training complete!");
//    }
//
//    public static void main(String[] args) {
//        try {
//            new NeuralNet3();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    /**
//     * Creates a matrix consisting of the row vector stacked on top of itself numRows times.
//     *
//     * @param row     the row vector
//     * @param numRows the number of times the row vector will be in the matrix
//     * @return the resulting matrix
//     */
//    @VisibleForTesting
//    static INDArray rowToMatrix(INDArray row, int numRows) {
//        INDArray base = row;
//        for (int i = 1; i < numRows; i++) {
//            base = Nd4j.vstack(base, row);
//        }
//
//        return base;
//    }
//
//    private static INDArray createPointsArray() {
//        int width = WebDriverExecutor.WINDOW_SIZE.width;
//        int height = WebDriverExecutor.WINDOW_SIZE.height;
//        int NUM_LOCS = 5;
//
//        ArrayList<INDArray> points = new ArrayList<>();
//
//        //top to right
//        for (int x = WebDriverExecutor.PIXELS_RIGHT + 10; x < width - WebDriverExecutor.PIXELS_LEFT - 10; x += width / NUM_LOCS) {
//            points.add(Nd4j.create(new float[]{(x + WebDriverExecutor.PIXELS_RIGHT), (50 + WebDriverExecutor.PIXELS_DOWN)}));
//        }
//
//        //left to bottom
//        for (int y = WebDriverExecutor.PIXELS_DOWN + 10; y < height - WebDriverExecutor.PIXELS_UP - 10; y += height / NUM_LOCS) {
//            points.add(Nd4j.create(new float[]{(50 + WebDriverExecutor.PIXELS_RIGHT), (y + WebDriverExecutor.PIXELS_DOWN)}));
//        }
//
//        //bottom to right
//        for (int x = WebDriverExecutor.PIXELS_RIGHT + 10; x < width - WebDriverExecutor.PIXELS_LEFT - 10; x += width / NUM_LOCS) {
//            points.add(Nd4j.create(new float[]{(x + WebDriverExecutor.PIXELS_RIGHT), (height - WebDriverExecutor.PIXELS_UP - 50)}));
//        }
//
//        //right to bottom
//        for (int y = WebDriverExecutor.PIXELS_DOWN + 10; y < height - WebDriverExecutor.PIXELS_UP - 10; y += height / NUM_LOCS) {
//            points.add(Nd4j.create(new float[]{(height - WebDriverExecutor.PIXELS_LEFT - 50), (y + WebDriverExecutor.PIXELS_DOWN)}));
//        }
//
//        return Nd4j.vstack(points);
//    }
//
//    public static MultiLayerNetwork customModel() {
//        /*
//          Use this method to build your own custom model.
//         */
//        return null;
//    }
//
////    public GameInstruction process(BufferedImage img) throws Exception {
////
////        INDArray imgArr = loader.asMatrix(img);
////
////        imgScaler.transform(imgArr);
////
////        int maxIndex = 0;
////        INDArray out = net.outputSingle(imgArr, pointsArr.getRow(0));
////        System.out.println("out = " + out);
////        double max = out.getDouble(0);
////
////        for (int i = 1; i < pointsArr.rows(); i++) {
////            double score = net.outputSingle(imgArr, pointsArr.getRow(i)).getDouble(0);
//////            System.out.println("score" + i + " " + score);
////            if (score > max) {
////                maxIndex = i;
////                max = score;
////            }
////        }
////
////        INDArray maxPoint = pointsArr.getRow(maxIndex).dup();
////
////        System.out.println("Max score: " + max);
////
////        pointScaler.revertFeatures(maxPoint);
////
////        return new GameInstruction(
////                new Point(maxPoint.getInt(0), maxPoint.getInt(1)), false);
////    }
////
////    public void trainState(GameState state) throws IOException {
////        INDArray imgArr = loader.asMatrix(state.img);
////
////        imgScaler.transform(imgArr);
////
////        System.out.println(imgArr.toString());
////
////        INDArray point = Nd4j.create(new float[]{state.mouseLoc.x, state.mouseLoc.y});
////
////        pointScaler.transform(point);
////
////        INDArray label = Nd4j.scalar(state.quality);
////        scoreScaler.transform(label);
////
////        System.out.println("training: " + state.quality + " transformed: " + label);
////
////        net.fit(new INDArray[]{imgArr, point}, new INDArray[]{label});
////    }
//
////    public static void main(String[] args) throws Exception {
////        new NeuralNet2().run(args);
////    }
//
//    //    public void run(String[] args) throws Exception {
////
////        new CudnnConvolutionHelper();
////
////
////        log.info("Load data....");
////
////
////        /*
////          Data Setup -> organize and limit data file paths:
////           - mainPath = path to image files
////           - fileSplit = define basic dataset split with limits on format
////           - pathFilter = define additional file load filter to limit size and balance batch content
////         */
////
////
//////        log.info("# Training examples: " + trainData.length());
//////        log.info("# Test examples: " + testData.length());
////
////
////        /*
////          Data Setup -> transformation
////           - Transform = how to tranform images and generate large dataset to train on
////         */
////        ImageTransform flipTransform1 = new FlipImageTransform(rng);
////        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
////        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
////        List<ImageTransform> transforms = Collections.emptyList();
////
////        /*
////          Data Setup -> normalization
////           - how to normalize images and generate large dataset to train on
////         */
////        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
////
////
////
////        log.info("Build model....");
////
////
////        /*
////          Data Setup -> define how to load data into net:
////           - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
////           - dataIter = a generator that only loads one batch at a time into memory to save memory
////           - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
////         */
////        MultipleEpochsIterator trainIter;
////
////
////        log.info("Train model....");
////        // Train without transformations
////
////        for (int i = 0; i < recordReader.getLabels().size(); i++) {
////            System.out.println(String.valueOf(i) + " - " + recordReader.getLabels().get(i));
////        }
////
////        scaler.fit(dataIter);
////        dataIter.setPreProcessor(scaler);
////        trainIter = new MultipleEpochsIterator(epochs, dataIter);
////        network.fit(trainIter);
////
////        // Train with transformations
////        for (ImageTransform transform : transforms) {
////            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
////            recordReader.initialize(trainData, transform);
////            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
////            scaler.fit(dataIter);
////            dataIter.setPreProcessor(scaler);
////            trainIter = new MultipleEpochsIterator(epochs, dataIter);
////            network.fit(trainIter);
////        }
////
////        log.info("Evaluate model....");
////        recordReader.initialize(testData);
////        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
////        scaler.fit(dataIter);
////        dataIter.setPreProcessor(scaler);
////        Evaluation eval = network.evaluate(dataIter);
////        log.info(eval.stats(true));
////
////        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
////        dataIter.reset();
////        DataSet testDataSet = dataIter.next();
////        List<String> allClassLabels = recordReader.getLabels();
////        int[] predictedClasses = network.predict(testDataSet.getFeatures());
////        String expectedResult = allClassLabels.get(labelIndex);
////        String modelPrediction = allClassLabels.get(predictedClasses[0]);
////
////        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");
////
////        if (save) {
////            log.info("Save model....");
////            String basePath = "D:\\Documents\\wormsAIModels\\";
////            ModelSerializer.writeModel(network, basePath + "model.bin", true);
////        }
////        log.info("****************Example finished********************");
////    }
////
////    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
////        return new ConvolutionLayer.Builder(kernel, stride, pad).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
////                .name(name).nIn(in).nOut(out).biasInit(bias).build();
////    }
////
////    private ConvolutionLayer conv3x3(String name, int out, double bias) {
////        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
////                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).name(name).nOut(out).biasInit(bias).build();
////    }
////
////    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
////        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad)
////                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).name(name).nOut(out).biasInit(bias).build();
////    }
////
////    private SubsamplingLayer maxPool(String name, int[] kernel) {
////        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
////    }
////
////    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
////        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
////    }
////
//    public ComputationGraph customNet() {
//        /*
//          AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
//          and the imagenetExample code referenced.
//          http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
//         */
//
//        double nonZeroBias = 1;
//        double dropOut = 0.5;
//
//        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder()
//                .seed(rng.nextLong())
//                .activation(Activation.RELU).weightInit(WeightInit.XAVIER)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 5e-4, 0.1, 100000), 0.7))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 6e-4, 0.1, 100000), 0.7))
//                .convolutionMode(ConvolutionMode.Same)
//                .graphBuilder()
//                .addInputs("image", "coordinates")
//                .addLayer("cnn1", new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
//                        .nIn(channels).nOut(20).activation(Activation.RELU).build(), "image")
//                .addLayer("maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
//                        new int[]{2, 2}).build(), "cnn1")
//                .addLayer("cnn2", new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
//                        .nOut(50).activation(Activation.RELU).build(), "maxpool1")
//                .addLayer("maxpool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
//                        new int[]{2, 2}).build(), "cnn2")
//                .addLayer("ffn1", new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build(),
//                        "maxpool2")
//                .addVertex("merge", new MergeVertex(), "ffn1", "coordinates")
//                .addLayer("ffn2", new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build(),
//                        "merge")
//                .addLayer("ffn3", new DenseLayer.Builder().activation(Activation.RELU).nOut(250).build(),
//                        "ffn2")
//                .addLayer("ffn4", new DenseLayer.Builder().activation(Activation.RELU).nOut(250).build(),
//                        "ffn3")
//                .addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                        .nOut(outputNum).activation(Activation.SOFTMAX).build(), "ffn4")
//                .setInputTypes(InputType.convolutional(height, width, channels), InputType.feedForward(2))
//                .setOutputs("out").build();
//
////        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder()
////                .seed(rng.nextLong()).weightInit(WeightInit.XAVIER)
////                .activation(Activation.RELU)
////                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
////                .updater(new Nesterovs(1e-3, 0.9))
//////                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-4, 0.1, 100000), 0.7))
////                .graphBuilder()
////                .addInputs("image", "coordinates")
////                .addLayer("ffn1", new DenseLayer.Builder().nOut(1000).build(),
////                        "image")
////                .addVertex("merge", new MergeVertex(), "ffn1", "coordinates")
////                .addLayer("ffn2", new DenseLayer.Builder().nOut(500).build(),
////                        "merge")
////                .addLayer("ffn3", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn2")
////                .addLayer("ffn4", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn3")
////                .addLayer("ffn5", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn4")
////                .addLayer("ffn6", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn5")
////                .addLayer("ffn7", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn6")
////                .addLayer("ffn8", new DenseLayer.Builder().nOut(500).build(),
////                        "ffn7")
////                .addLayer("ffn9", new DenseLayer.Builder().nOut(250).build(),
////                        "ffn8")
////                .addLayer("ffn10", new DenseLayer.Builder().nOut(100).build(),
////                        "ffn9")
////                .addLayer("ffn11", new DenseLayer.Builder().nOut(50).build(),
////                        "ffn10")
////                .addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
////                        .nOut(outputNum).activation(Activation.SOFTMAX).build(), "ffn11")
////                .setInputTypes(InputType.convolutional(height, width, channels), InputType.feedForward(2))
////                .setOutputs("out").build();
//
//        return new ComputationGraph(conf1);
//
////        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rng.nextLong())
////                .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
////                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta())
////                .convolutionMode(ConvolutionMode.Same).list()
////                // block 1
////                .layer(new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}).name("cnn1")
////                        .nIn(channels).nOut(20).activation(Activation.RELU).build())
////                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
////                        new int[]{2, 2}).name("maxpool1").build())
////                // block 2
////                .layer(new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}).name("cnn2").nOut(50)
////                        .activation(Activation.RELU).build())
////                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
////                        new int[]{2, 2}).name("maxpool2").build())
//////            .layer(4, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}).name("cnn3").nOut(50)
//////                .activation(Activation.RELU).build())
//////            .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
//////                new int[]{2, 2}).name("maxpool3").build())
////                // fully connected
////                .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(500).build())
////                // output
////                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
////                        .nOut(outputNum).activation(Activation.SOFTMAX) // radial basis function required
////                        .build())
////                .setInputType(InputType.convolutionalFlat(height, width, channels))
////                .backprop(true).pretrain(false).build();
//
//
//    }
//
//}
