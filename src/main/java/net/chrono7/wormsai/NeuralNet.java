package net.chrono7.wormsai;

//import org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;

public class NeuralNet {

    private MultiLayerNetwork net = null;
    private NativeImageLoader loader = new NativeImageLoader(200, 200, 3);

    public NeuralNet () throws IOException {

        File netSaveFile = new File("D:\\Documents\\wormsAIModels\\model.bin");
        net = ModelSerializer.restoreMultiLayerNetwork(netSaveFile);

    }

    public BlobType process(Mat mat_in) throws Exception {
//        DataSet ds = new DataSet();

        org.bytedeco.javacpp.opencv_core.Mat mat = new org.bytedeco.javacpp.opencv_core.Mat((Pointer)null)
        { { address = mat_in.getNativeObjAddr(); } };

        INDArray arr = loader.asMatrix(mat);

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
