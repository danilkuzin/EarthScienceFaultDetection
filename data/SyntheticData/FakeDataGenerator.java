package ipf;

import edu.mines.jtk.mosaic.PixelsView;
import edu.mines.jtk.mosaic.SimplePlot;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import static edu.mines.jtk.util.ArrayMath.*;
import static edu.mines.jtk.util.ArrayMath.fillfloat;
import static ipf.FakeData.seismicAndSlopes2d2014A;

public class FakeDataGenerator extends FakeData {
    public static void main(final String[] args) {
        SwingUtilities.invokeLater(new Runnable(){
            public void run() {
                go(args);
            }
        });
    }

    private static int toRGB(float value) {
        int part = Math.round(value * 255);
        return part * 0x10101;
    }

    private static void go(String[] args) {
        int insize = 600;
        if (args.length == 0 || args[0].equals("seismicAndSlopes2d2014A")) {
            for (long s = 1; s < 10; s++) {
                float[][][] gp = dataForGeologyFault(0.0, s, insize);
                float[][] g = gp[0];
                //float[][] p = gp[1];
                //SimplePlot sp = new SimplePlot(SimplePlot.Origin.UPPER_LEFT);
                //PixelsView pv = sp.addPixels(g);
                //sp.setSize(700, 700);
                BufferedImage image = new BufferedImage(insize, insize, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < insize; y++)
                    for (int x = 0; x < insize; x++)
                        image.setRGB(x, y, toRGB(g[y][x]));

                try {
                    ImageIO.write(image, "png", new File("/home/olga/PycharmProjects/EarthScienceFaultDetection/data/SyntheticData/test"+s+".png"));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static float[][][] dataForGeologyFault(double noise, long seed, int imsize) {
        int n1 = imsize;
        int n2 = imsize;
        float[][][] p = makeReflectivityWithNormalsSeed(n1,n2, seed);
        float[][][] q = makeReflectivityWithNormalsSeed(n1,n2, seed);
        float[][][] r = makeReflectivityWithNormalsSeed(n1,n2, seed);
        Random random = new Random(seed);
        float[] slope = randfloat(random, 2);
        FakeData.Linear1 throw1 = new FakeData.Linear1(slope[0],slope[1]);
        FakeData.Linear1 throw2 = new FakeData.Linear1(0.0f,0.10f);
        float[] theta = randfloat(random, 1);
        FakeData.LinearFault2 fault1 = new FakeData.LinearFault2(0.0f,n2*0.2f, 100*theta[0],throw1);
        FakeData.LinearFault2 fault2 = new FakeData.LinearFault2(0.0f,n2*0.4f,-15.0f,throw2);
        FakeData.Sinusoidal2 fold = new FakeData.Sinusoidal2(0.04f,0.05f,1.9e-4f,2.0e-4f);
        FakeData.VerticalShear2 shear = new FakeData.VerticalShear2(new FakeData.Linear1(0.0f,0.05f));
        p = apply(fold,p);
        p = combine(n1/3,q,p);
        //p = apply(shear,p);
        //p = combine(n1/6,r,p);
        p = apply(fault1,p);
        p = apply(fault2,p);
        p = addWavelet(0.1,p);
        p[0] = addNoise(noise,p[0]);
        p[1] = neg(div(p[2],p[1]));
        return new float[][][]{p[0],p[1]};
    }

    protected static float[][][] makeReflectivityWithNormalsSeed(int n1, int n2, long seed) {
        Random random = new Random(seed);
        float[] r = pow(mul(2.0f,sub(randfloat(random,n1),0.5f)),5.0f);
        //float[] r = mul(2.0f,sub(randfloat(random,n1),0.5f));
        float[][][] p = new float[3][n2][n1];
        for (int i2=0; i2<n2; ++i2)
            copy(r,p[0][i2]);
        p[1] = fillfloat(1.0f,n1,n2);
        p[2] = fillfloat(0.0f,n1,n2);
        return p;
    }

}
