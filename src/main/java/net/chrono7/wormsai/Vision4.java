package net.chrono7.wormsai;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Vision4 {
    private static final double THRESHOLD = 35;
    private static Java2DFrameConverter java2DFrameConverter = new Java2DFrameConverter();
    private static OpenCVFrameConverter.ToMat openCVFrameConverter = new OpenCVFrameConverter.ToMat();
    private static Method fillNDArray = null;

    public static Pair<Integer, INDArray> processAndCountLargeBlobs(BufferedImage img) throws IOException {

        INDArray arr = NeuralNet4.loader.asMatrix(img);
        NeuralNet4.scaler.transform(arr);

        return new Pair<>(countLargeBlobs(img), arr);
    }

    private static int countLargeBlobs(BufferedImage img) {

        Mat m = img2Mat(img);

        cvtColor(m, m, COLOR_BGR2GRAY);

        threshold(m, m, THRESHOLD, 255, THRESH_BINARY);

        opencv_core.Point2f center = new opencv_core.Point2f(m.cols() / 2, m.rows() / 2);

        opencv_core.MatVector contours = new opencv_core.MatVector();
        findContours(m, contours, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);

        int large = 0;

        for (int i = 0; i < contours.size(); i++) {
            Mat c = contours.get(i);

            double dstToCenter = pointPolygonTest(c, center, true);
            dstToCenter = dstToCenter >= 0 ? 0 : Math.abs(dstToCenter);

            if (dstToCenter > 0) {
                if (contourArea(c) > 5000) {
                    large++;
                }
            }

        }

        return large;
    }


    /**
     * Performs unsafe conversion of a Mat of the correct size to an INDArray.
     * Size needs to match {@link NeuralNet4#WIDTH} and {@link NeuralNet4#HEIGHT} and channels need to be
     * {@link NeuralNet4#CHANNELS}. Uses reflection and should be faster than {@link NativeImageLoader#asMatrix(Mat)}
     *
     * @param m the mat to convert
     * @return the resulting INDArray
     */
    @Deprecated
    public static INDArray process(Mat m) {

        if (fillNDArray == null) {
            Class loader = NativeImageLoader.class;
            try {
                fillNDArray = loader.getDeclaredMethod("fillNDArray", Mat.class, INDArray.class);
                fillNDArray.setAccessible(true);
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
        }

//         INDArray arr = NeuralNet4.loader.asMatrix(m);
        INDArray ret = Nd4j.create(m.channels(), m.rows(), m.cols());

        try {
            fillNDArray.invoke(NeuralNet4.loader, m, ret);
            m.data(); // dummy call to make sure it does not get deallocated prematurely
        } catch (IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }

        ret.reshape(ArrayUtil.combine(new int[][]{{1}, ret.shape()}));

        NeuralNet4.scaler.transform(ret);

        return ret;
    }


//    public static INDArray process(Mat m) {
//
//        try {
//            INDArray arr = NeuralNet4.loader.asMatrix(m);
//            NeuralNet4.scaler.transform(arr);
//            return arr;
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//
//        return null;
//    }

    @Deprecated
    public static Mat shrink(Frame frame) {
        if (frame == null) {
            return null;
        }

        Mat m = openCVFrameConverter.convert(frame);
        Mat m2 = new Mat();
        opencv_imgproc.resize(m, m2, new opencv_core.Size(NeuralNet4.WIDTH, NeuralNet4.HEIGHT));
//        threshold(m2, m2, THRESHOLD, 255, THRESH_BINARY);
        if (NeuralNet4.CHANNELS == 1) {
            cvtColor(m2, m2, CV_BGR2GRAY);
        } else if (NeuralNet4.CHANNELS != 3) {
            throw new RuntimeException("Channels should be 1 or 3!");
        }

        return m2;
    }

    public static INDArray convert(Frame frame) {
        if (frame == null) {
            return null;
        }

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WorkspaceManager.CPU_ID)) {
                INDArray array = NeuralNet4.loader.asMatrix(frame);
                NeuralNet4.scaler.transform(array);
                return array;
                //TODO: assert attached

            } catch (IOException e) {
                e.printStackTrace();
            }

        return null;
    }

    private static Mat img2Mat(BufferedImage in) {
        return openCVFrameConverter.convert(java2DFrameConverter.convert(in));
    }
}
