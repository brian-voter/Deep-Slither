package net.chrono7.wormsai;

import org.bytedeco.javacpp.Pointer;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class NeuralNet {

    private MultiLayerNetwork net = null;
    private ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    private NativeImageLoader loader = new NativeImageLoader(150, 150, 3);
    private List<String> labelNames = Arrays.asList("FOOD", "PREY", "WORM");

    public NeuralNet() throws IOException {

        File netSaveFile = new File("D:\\Documents\\wormsAIModels\\model.bin");
        net = ModelSerializer.restoreMultiLayerNetwork(netSaveFile);


    }

    public void process(List<Blob> blobs) throws Exception {

        if (blobs.size() == 0) {
            return;
        }

//        DataSet ds = new DataSet();
//        ds.setLabelNames(labelNames);

        for (Blob b : blobs) {
            org.bytedeco.javacpp.opencv_core.Mat mat = new org.bytedeco.javacpp.opencv_core.Mat((Pointer) null) {
                {
                    address = b.mat.getNativeObjAddr();
                }
            };


            INDArray arr = loader.asMatrix(mat);
            scaler.transform(arr);
//            System.out.println(arr);
//            ds.addFeatureVector(arr);

//            System.out.println(Arrays.toString(net.predict(arr)));
            b.blobType = BlobType.fromInt(net.predict(arr)[0]);
            System.out.println(b.blobType);
        }

//        DataSetIterator it = new ViewIterator(ds, 7);

//        net.labelProbabilities(ds.getFeatures());

//        List<String> predictions = net.predict(ds);
//
//        for (int i = 0; i < predictions.size(); i++) {
//            blobs.get(i).blobType = BlobType.valueOf(predictions.get(i));
//        }

    }

    public BlobType process(Mat mat_in) throws Exception {
//        DataSet ds = new DataSet();

        org.bytedeco.javacpp.opencv_core.Mat mat = new org.bytedeco.javacpp.opencv_core.Mat((Pointer) null) {
            {
                address = mat_in.getNativeObjAddr();
            }
        };

        INDArray arr = loader.asMatrix(mat);
        scaler.transform(arr);

//        System.out.println(arr.shapeInfoToString());
//        ds.addFeatureVector(arr);

        int[] predictions = net.predict(arr);

        return BlobType.fromInt(predictions[0]);
    }

//    public void process(opencv_core.Mat[] mats) throws IOException {
//        DataSet ds = new DataSet();
//
//        for (opencv_core.Mat m : mats) {
//            ds.addFeatureVector(loader.asMatrix(m));
//        }
//
//        DataSetIterator it = new TestDataSetIterator()
//    }

}
