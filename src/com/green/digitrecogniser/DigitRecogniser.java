package com.green.digitrecogniser;

import com.green.nn.NeuralNetwork;
import com.green.nn.Neuron;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DigitRecogniser {
    private final String TRAINING_FILE;
    private BufferedReader reader;
    private NeuralNetwork network;

    public DigitRecogniser(String trainingFile, String testFile) {
        TRAINING_FILE = trainingFile;
        reader = null;
        network = new NeuralNetwork(784, 10, 1);
    }

    private List<Double> readLine() {
        List<Double> parsedValues = new ArrayList<>();
        try {
            String line = reader.readLine();
            if(line == null) return null;

            String[] lineSplit = line.split(",");
            for(int i=0;i<lineSplit.length;i++) {
                parsedValues.add(Double.parseDouble(lineSplit[i]));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return parsedValues;
    }

    private void openFile(String filename) {
        try {
            reader = new BufferedReader(new FileReader(filename));

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private void closeFile() {
        try {
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void train() {
        openFile(TRAINING_FILE);
        List<Double> input;
        int cMiss = 0;
        int counter = 0;
        while((input = readLine()) != null) {
            int label = input.get(0).intValue();
            input.remove(0);

            network.setInputValues(input);

            List<Double> output = network.compute();

            counter++;
            if(label == 1 && network.thresholdFunction(output.get(0)) != 1) {
                network.outputs.get(0).adjustWeights(1);

                for(Neuron n: network.hidden){
                    n.adjustWeights(network.outputs);
                }
                cMiss++;
            }
            System.out.println(label + " :: " + output.get(0));
            //System.out.println("Misses: " + cMiss + " :: Total: " +counter+" :: ratio: "+(1.0*cMiss)/counter);

        }

        closeFile();
    }

    public static void main(String[] args) {
        DigitRecogniser dr = new DigitRecogniser("/Users/wgreen/Documents/wspace/JANN/src/train_noh.csv", null);
        dr.train();
    }
}
