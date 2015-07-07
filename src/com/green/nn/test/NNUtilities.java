package com.green.nn.test;

import java.util.Random;

public class NNUtilities {
    public static Random rand = new Random();

    /**
     * Generates a random double between -1 and 1
     * @return
     */
    public static double randomDouble() {
        return (rand.nextInt(180) - 90.0) / 100.0;
    }
}
