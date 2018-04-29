package net.chrono7.wormsai;

import org.opencv.core.*;
import org.opencv.img_hash.ColorMomentHash;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

public class Vision3 {

    private static final int ERODE_SIZE = 3;
    private static final int OPENING_SIZE = 20;
    private static final int CLOSING_SIZE = 10;
    private static final double THRESHOLD = 35.0;
    private Point center;

    private Mat erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ERODE_SIZE, ERODE_SIZE));
    private Mat openingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(OPENING_SIZE, OPENING_SIZE));
    private Mat closingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(CLOSING_SIZE, CLOSING_SIZE));

    private NeuralNet net = new NeuralNet();

    public Vision3() throws IOException {

    }

    public static void main(String[] args) throws IOException {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Vision3 vision = new Vision3();

        ColorMomentHash h = ColorMomentHash.create();

        Mat m1 = Imgcodecs.imread("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\out22.png");
        Mat m2 = Imgcodecs.imread("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\out24.png");

        Mat m11 = new Mat(m1.rows(), m1.cols(), CvType.CV_8UC3, Scalar.all(0));
        Mat m21 = new Mat(m1.rows(), m1.cols(), CvType.CV_8UC3, Scalar.all(0));

        h.compute(m1, m11);
        h.compute(m2, m21);

        System.out.println("diff: " + h.compare(m11, m21));

        Imgcodecs.imwrite("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\out2.png",
                vision.process(Imgcodecs.imread("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\out.png")).mat);

    }

    /**
     * Credit: https://stackoverflow.com/a/42028517
     */
    protected Mat img2Mat(BufferedImage in) {
        Mat out;
        byte[] data;
        int r, g, b;

        if (in.getType() == BufferedImage.TYPE_INT_RGB) {
            out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC3);
            data = new byte[in.getWidth() * in.getHeight() * (int) out.elemSize()];
            int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null, 0, in.getWidth());
            for (int i = 0; i < dataBuff.length; i++) {
                data[i * 3] = (byte) ((dataBuff[i] >> 0) & 0xFF);
                data[i * 3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
                data[i * 3 + 2] = (byte) ((dataBuff[i] >> 16) & 0xFF);
            }
        } else {
            out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC1);
            data = new byte[in.getWidth() * in.getHeight() * (int) out.elemSize()];
            int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null, 0, in.getWidth());
            for (int i = 0; i < dataBuff.length; i++) {
                r = (byte) ((dataBuff[i] >> 0) & 0xFF);
                g = (byte) ((dataBuff[i] >> 8) & 0xFF);
                b = (byte) ((dataBuff[i] >> 16) & 0xFF);
                data[i] = (byte) ((0.21 * r) + (0.71 * g) + (0.07 * b));
            }
        }
        out.put(0, 0, data);
        return out;
    }

    public VisionState process(BufferedImage img) {
        return process(img2Mat(img));
    }

    public VisionState process(Mat m_in) {

        Mat m = m_in.clone();

        if (center == null) {
            center = new Point(m.width() / 2, m.height() / 2);
        }


//        Mat m = Imgcodecs.imread("C:\\Users\\Brian\\IdeaProjects\\WormsAI\\out.png");

        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2GRAY);

        Imgproc.threshold(m, m, THRESHOLD, 255, Imgproc.THRESH_BINARY);

        //TODO: Can remove food with "opening" = erosion, dilate

//        Imgproc.erode(m, m, erodeKernel);

//        Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, openingKernel);
//        Imgproc.morphologyEx(m, m, Imgproc.MORPH_ERODE, erodeKernel);

        List<MatOfPoint> contours = new ArrayList<>();

        Mat hierarchy = new Mat();

        Imgproc.findContours(m, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_KCOS);

        Mat out = new Mat(m.rows(), m.cols(), CvType.CV_8UC3);
        out.setTo(Scalar.all(0));

        AtomicReference<Blob> self = new AtomicReference<>();
        List<MatOfPoint> selfContour = new ArrayList<>();


        List<MatOfPoint> wormsMats = new ArrayList<>();
        List<MatOfPoint> foodMats = new ArrayList<>();
        List<MatOfPoint> preyMats = new ArrayList<>();
        List<Blob> worms = new ArrayList<>();
        List<Blob> food = new ArrayList<>();
        List<Blob> prey = new ArrayList<>();
        List<Blob> unknown = new ArrayList<>();

        List<MatOfPoint> wormsPolys = new ArrayList<>();

        for (MatOfPoint c : contours) {
            MatOfPoint2f m2 = new MatOfPoint2f();
            c.convertTo(m2, CvType.CV_32F);

            double dstToCenter = Imgproc.pointPolygonTest(m2, center, true);

            //Set to 0 if it the center mouseLoc of the image is on or inside the polygon
            dstToCenter = dstToCenter >= 0 ? 0 : Math.abs(dstToCenter);

            Blob b = new Blob(c, m2, dstToCenter, Imgproc.contourArea(c), BlobType.UNKNOWN);

            if (b.distanceFromCenter > 0) {
                // This contour is not the self

                // All worms seem to have an area of ~6000. Filters out food.
                if (Imgproc.contourArea(b.contour) > 5500) {

                    Point blobCenter = new Point();
                    float[] radiusArr = new float[1];

                    Imgproc.minEnclosingCircle(b.mp2f, blobCenter, radiusArr);

                    int radius = (int) radiusArr[0];

                    int circleArea = (int) (Math.PI * radius * radius);

                    //ratio of contour area to minimum enclosing circle area
                    float extent = ((float) b.area) / circleArea;

                    //likey big blob
//                    if (extent > 0.6) {
//                        food.add(b);
//                    } else {

                        b.mat = captureBlob(m_in, b);
                        unknown.add(b);

//                        b.blobType = captureBlob(m_in, b);
//
//                        if (b.blobType == BlobType.WORM) {
//                            wormsMats.add(b.contour);
//                            worms.add(b);
//                        } else {
//                            food.add(b);
//                        }

                        //ROI credit: http://answers.opencv.org/question/497/extract-a-rotatedrect-area/?answer=518#post-id-518

//                    }

                } else {
                    //Save this blob
                    food.add(b);
                }
            } else {
                //This worm is the self
                selfContour.add(b.contour);
                self.set(b);
            }
        }

        try {
            net.process(unknown);
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (Blob next : unknown) {
            switch (next.blobType) {
                case WORM:
                    worms.add(next);
                    wormsMats.add(next.contour);
//                    System.out.println("WORM IT");
                    break;
                case FOOD:
                    foodMats.add(next.contour);
                    food.add(next);
//                    System.out.println("FOOD IT");
                    break;
                case PREY:
//                    System.out.println("PREY IT");
                    prey.add(next);
                    preyMats.add(next.contour);
                    break;
            }
        }

        if (self.get() == null) {
            if (worms.size() > 0) {
                worms.sort(Comparator.comparingDouble(b -> b.distanceFromCenter));
                self.set(worms.get(0));
                worms.remove(self.get());
                selfContour.add(self.get().contour);
            }
        }

        try {
            Imgproc.fillPoly(out, foodMats, Scalar.all(255));
        } catch (NullPointerException e) {
            e.printStackTrace();
        }

        try {
            Imgproc.fillPoly(out, wormsMats, new Scalar(0, 0, 255));
        } catch (NullPointerException e) {
            e.printStackTrace();
        }

        try {
            Imgproc.fillPoly(out, preyMats, new Scalar(0, 255, 0));
        } catch (NullPointerException e) {
            e.printStackTrace();
        }

        try {
            Imgproc.fillPoly(out, selfContour, new Scalar(255, 0, 0));
        } catch (NullPointerException e) {
            e.printStackTrace();
        }

        return new VisionState(out, worms, food, prey, self.get(), System.currentTimeMillis());
    }

    private Mat captureBlob(Mat m_in, Blob b) {
        RotatedRect rect = Imgproc.minAreaRect(b.mp2f);
        // matrices we'll use
        Mat rotMatrix, rotated = new Mat(), cropped = new Mat();
        // get angle and size from the bounding box
        double angle = rect.angle;
        Size rect_size = rect.size;

        if (rect.size.height > rect.size.width) {
            angle += 90.0;

            //swap
            double temp = rect_size.width;

            //noinspection SuspiciousNameCombination
            rect_size.width = rect_size.height;
            rect_size.height = temp;
        }
        // get the rotation matrix
        rotMatrix = Imgproc.getRotationMatrix2D(rect.center, angle, 1.0);

        Mat blobMask = new Mat(m_in.rows(), m_in.cols(), m_in.type(), Scalar.all(0));

        ArrayList<MatOfPoint> blobPts = new ArrayList<>(1);
        blobPts.add(b.contour);

        Imgproc.fillPoly(blobMask, blobPts, Scalar.all(255));

        Mat blobOut = new Mat(m_in.rows(), m_in.cols(), m_in.type(), Scalar.all(0));
        Core.bitwise_and(m_in, blobMask, blobOut);

        // perform the affine transformation
        Imgproc.warpAffine(blobOut, rotated, rotMatrix, m_in.size(), Imgproc.INTER_CUBIC);
        // crop the resulting image
        Imgproc.getRectSubPix(rotated, rect_size, rect.center, cropped);

//        BlobType result = BlobType.UNKNOWN;
//
//        try {
//            result = net.process(cropped);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

//        Imgcodecs.imwrite("D:\\Documents\\wormImages\\" +
//                String.valueOf(System.currentTimeMillis()) + ".png", cropped);

        return cropped;

    }

    private MatOfPoint2f getConvexHull(Blob b) {
        MatOfInt convexHullIndicies = new MatOfInt();
        Imgproc.convexHull(b.contour, convexHullIndicies);
        Point[] contourPoints = b.contour.toArray();
        List<Point> convexPoints = new ArrayList<>();
        for (int i : convexHullIndicies.toArray()) {
            convexPoints.add(contourPoints[i]);
        }
        Point[] convexArr = convexPoints.toArray(new Point[0]);

        return new MatOfPoint2f(convexArr);
    }

    private MatOfInt getConvexHullIndicies(Blob b) {
        MatOfInt convexHullIndicies = new MatOfInt();
        Imgproc.convexHull(b.contour, convexHullIndicies);

        return convexHullIndicies;
    }
}