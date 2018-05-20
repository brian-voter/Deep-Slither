package net.chrono7.deepslither;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.awt.image.BufferedImage;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

/**
 * @author Brian Voter
 */
public class Vision {
//    private static final double THRESHOLD = 35;
    private static Java2DFrameConverter java2DFrameConverter = new Java2DFrameConverter();
    private static OpenCVFrameConverter.ToMat openCVFrameConverter = new OpenCVFrameConverter.ToMat();
    private static Method fillNDArray = null;

    /**
     * Performs unsafe conversion of a Mat of the correct size to an INDArray.
     * Size needs to match {@link NeuralNet#WIDTH} and {@link NeuralNet#HEIGHT} and channels need to be
     * {@link NeuralNet#CHANNELS}. Uses reflection and should be faster than {@link NativeImageLoader#asMatrix(Mat)}
     *
     * @param m the mat to frame2INDArray
     * @return the resulting INDArray
     */
    public static INDArray mat2INDArray(Mat m) {

        if (fillNDArray == null) {
            Class loader = NativeImageLoader.class;
            try {
                fillNDArray = loader.getDeclaredMethod("fillNDArray", Mat.class, INDArray.class);
                fillNDArray.setAccessible(true);
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
        }

        INDArray ret = Nd4j.create(m.channels(), m.rows(), m.cols());

        try {
            fillNDArray.invoke(NeuralNet.loader, m, ret);
            m.data(); // dummy call to make sure it does not get deallocated prematurely
        } catch (IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }

        ret.reshape(ArrayUtil.combine(new int[][]{{1}, ret.shape()}));

        NeuralNet.scaler.transform(ret);

        return ret;
    }

    public static Mat frame2Mat(Frame frame) {
        if (frame == null) {
            return null;
        }

        Mat m = openCVFrameConverter.convert(frame);

        Mat m2 = new Mat();
        opencv_imgproc.resize(m, m2, new opencv_core.Size(NeuralNet.WIDTH, NeuralNet.HEIGHT));
//        threshold(m2, m2, THRESHOLD, 255, THRESH_BINARY);
        if (NeuralNet.CHANNELS == 1) {
            cvtColor(m2, m2, CV_BGR2GRAY);
        } else if (NeuralNet.CHANNELS != 3) {
            throw new RuntimeException("Channels should be 1 or 3!");
        }

        return m2;
    }

    public static INDArray frame2INDArray(Frame frame) {
        if (frame == null) {
            return null;
        }

        Mat m = frame2Mat(frame);
        return mat2INDArray(m);
    }

    private static Mat img2Mat(BufferedImage in) {
        return openCVFrameConverter.convert(java2DFrameConverter.convert(in));
    }
}
