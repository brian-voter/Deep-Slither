package net.chrono7.wormsai;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Vision {

    private static final float SIMILAR_HUE_THRESHOLD = 200F;
    private static int w, h;

    /**
     * @param img  The image to filter
     * @param step The step size for downscaling
     * @return returns a color array with dimensions (width / step, height / step). Colors are
     * in 3-byte format RGB or BGA
     */
//    public static int[][] filterImage(BufferedImage img, final int step) {
//        final int threshold = 75;
//
//        final int w = img.getWidth();
//        final int h = img.getHeight();
//        final int[][] map = new int[w / step][h / step];
//        for (int x = 0, xc = 0; x < w; x += step, xc++) {
//            for (int y = 0, yc = 0; y < h; y += step, yc++) {
//                int c = img.getRGB(x, y);
//                if (!((c & 0xFF) <= threshold &&
//                        (c & 0xFF00) <= (threshold << 8) &&
//                        (c & 0xFF0000) <= (threshold << 16))) {
//                    map[x / step][y / step] = c;
//                }
//            }
//        }
//
//        System.out.println("ORIG W, H: " + img.getWidth() + ", " + img.getHeight() + " NEW W, H " + w + ", " + h);
//
//        return map;
//    }

    public static int[][] filterImage(BufferedImage img, final int step, int[][] map) {
        if (w == 0){
            w = img.getWidth();
            h = img.getHeight();
        }

        final int threshold = 100;
        if (map == null){
            map = new int[w / step + 1][h / step + 1];
        }
        for (int x = 0, xc = 0; x < w; x += step, xc++) {
            for (int y = 0, yc = 0; y < h; y += step, yc++) {
                int c = img.getRGB(x, y);
                if (!((c & 0xFF) <= threshold &&
                        (c & 0xFF00) <= (threshold << 8) &&
                        (c & 0xFF0000) <= (threshold << 16))) {
                    //map[x/step + w*y/step/step] = c;
                    map[xc][yc] = c;
                }
            }
        }

        return map;
    }


    /**
     * Gets the color distance between two colors in RGB format using CIELAB
     * @param c1
     * @param c2
     * @return
     */
    private static double getColorDistance(Color c1, Color c2)
    {
        float[] c1LAB = c1.getColorComponents(CIELab.getInstance(), null);
        float[] c2LAB = c2.getColorComponents(CIELab.getInstance(), null);

        double dist =  Math.sqrt(Math.pow(c1LAB[0] - c2LAB[0], 2) + Math.pow(c1LAB[1] - c2LAB[1], 2) + Math.pow(c1LAB[2] - c2LAB[2], 2));
//        System.out.println("DIST: " + dist);
        return dist;
    }

    /**
     * Compares two colors in int rgb format
     * @param color1
     * @param color2
     * @return
     */
    public static boolean areColorsSimilar(int color1, int color2) {
        return areColorsSimilar(new Color(color1), new Color(color2));
    }

    /**
     * Compares two colors.
     * @param c1
     * @param c2
     * @return
     */
    public static boolean areColorsSimilar(Color c1, Color c2){
        return getColorDistance(c1, c2) < SIMILAR_HUE_THRESHOLD;
    }
//
//    public static boolean areColorsSimilar(float[] color1, float[] color2) {
//        return getColorDistance(color1, color2) < SIMILAR_HUE_THRESHOLD;
//    }

//    /**
//     * Gets the HSB Hue component. See {@link Color#RGBtoHSB(int, int, int, float[])}.
//     *
//     * @param r The red component
//     * @param g The green component
//     * @param b The blue component
//     * @return The Hue component from the HSB colorspace.
//     */
//    private static float getHSBHue(int r, int g, int b) {
//        float hue, saturation;
//        int cmax = (r > g) ? r : g;
//        if (b > cmax) cmax = b;
//        int cmin = (r < g) ? r : g;
//        if (b < cmin) cmin = b;
//
//        if (cmax != 0)
//            saturation = ((float) (cmax - cmin)) / ((float) cmax);
//        else
//            saturation = 0;
//        if (saturation == 0)
//            hue = 0;
//        else {
//            float redc = ((float) (cmax - r)) / ((float) (cmax - cmin));
//            float greenc = ((float) (cmax - g)) / ((float) (cmax - cmin));
//            float bluec = ((float) (cmax - b)) / ((float) (cmax - cmin));
//            if (r == cmax)
//                hue = bluec - greenc;
//            else if (g == cmax)
//                hue = 2.0f + redc - bluec;
//            else
//                hue = 4.0f + greenc - redc;
//            hue = hue / 6.0f;
//            if (hue < 0)
//                hue = hue + 1.0f;
//        }
//
//        return hue;
//    }
}
