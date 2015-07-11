package com.green.digitrecogniser;

import com.green.nn.NeuralNetwork;
import com.green.nn.Neuron;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Trains a neural network to recognize handwritten digits
 * 
 * @author wsgreen
 *
 */
public class DigitRecogniser {
    private final String TRAINING_FILE;
    private BufferedReader reader;
    private NeuralNetwork network;

    public DigitRecogniser(String trainingFile) {
        TRAINING_FILE = trainingFile;
        reader = null;
        network = new NeuralNetwork(784, 15, 10);
    }

    /**
     * Read currently open file
     * 
     * @return
     */
    private List<Double> readLine() {
        List<Double> parsedValues = new ArrayList<>();
        try {
            String line = reader.readLine();
            if(line == null) return null;

            String[] lineSplit = line.split(",");
            for(int i=0;i<lineSplit.length;i++) {
                parsedValues.add(Double.parseDouble(lineSplit[i])/255.000);
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
    
    /**
     * Test if output is correct
     * 
     * @param label
     * @param outputs
     * @return
     */
    private boolean testOutput(int label, List<Integer> outputs) {
    	if(outputs.get(label) != 1)
    		return false;
    	
    	for(int i=0;i<outputs.size();i++) {
    		if(i!=label && outputs.get(i)==1)
    			return false;
    	}
    	
    	return true;		
    }
    
    /**
     * Create list of expected values for output nodes
     * 
     * @param label
     * @return
     */
    private List<Integer> getExpected(int label) {
    	List<Integer> expected = new ArrayList<>(Collections.nCopies(10, 0));
    	expected.add(label, 1);
    	return expected;
    }

    /**
     * Read training file and adjust weights 
     */
    public void train() {
        openFile(TRAINING_FILE);
        List<Double> input;
        int cMiss = 0;
        int counter = 0;
        while((input = readLine()) != null) {
            int label =(int) (input.get(0)*255);
            input.remove(0);
            
            network.setInputValues(input);
            List<Integer> output = network.compute();
            if(!testOutput(label, output)){
            	network.train(getExpected(label));
            	cMiss++;
            }
            
            counter++;
            System.out.println("Misses: " + cMiss + " :: Total: " +counter+" :: ratio: "+(1.0*cMiss)/counter);

        }



        closeFile();
    }
    
    public void test() {

            openFile(TRAINING_FILE);
            List<Double> input;
            int cMiss = 0;
            int counter = 0;
            while((input = readLine()) != null) {
                int label =(int) (input.get(0)*255);
                input.remove(0);

                network.setInputValues(input);

                List<Integer> output = network.compute();

                counter++;
                if(!testOutput(label, output)) {
                    cMiss++;
                }
            }
            System.out.println("++++++++++++++++++++++++++");
            System.out.println("Misses: " + cMiss + " :: Total: " +counter+" :: ratio: "+(1.0*cMiss)/counter);



            closeFile();
 
    }
    
    public static void main(String[] args) {
        DigitRecogniser dr = new DigitRecogniser("/Users/wsgreen/Documents/workspace/JANN/src/train.csv");
        dr.train();
        dr.train();
        dr.test();
    }
}
