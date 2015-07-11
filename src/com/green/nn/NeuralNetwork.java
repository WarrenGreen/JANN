package com.green.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Neural Network supporting one hidden layer and threshold output
 * 
 * @author wsgreen
 *
 */
public class NeuralNetwork {
	private List<Neuron> inputs;
	private List<Neuron> hidden;
	private List<Neuron> outputs;
	public static double learningRate = 1.75;
	
	/**
	 * 
	 * @param inputSize
	 * @param hiddenSize
	 * @param outputSize
	 */
	public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
		inputs = new ArrayList<>();
		hidden = new ArrayList<>();
		outputs = new ArrayList<>();
		
		for(int i=0;i<inputSize;i++)
			inputs.add(new Neuron());
		
		for(int i=0;i<hiddenSize;i++) {
			Neuron n = new Neuron();
			n.addAxons(inputs);
			hidden.add(n);
		}
		
		for(int i=0;i<outputSize;i++) {
			Neuron n = new Neuron();
			n.addAxons(hidden);
			outputs.add(n);
		}

	}
	
	/**
	 * Compute values for each node from hidden to output
	 * @return
	 */
	public List<Integer> compute() {
		List<Integer> outputValues = new ArrayList<>();
		
		for(Neuron n: hidden)
			n.compute();
		
		for(Neuron n: outputs) {
			n.compute();
			outputValues.add(thresholdFunction(n.getValue()));
		}
		
		return outputValues;
	}
	
	/**
	 * Adjust weights for each node based on expected outputs for output nodes
	 * @param expected
	 */
	public void train(List<Integer> expected) {
		for(int i=0;i<outputs.size();i++)
			outputs.get(i).adjustWeights(expected.get(i));
		
		for(Neuron n: hidden)
			n.adjustWeights(outputs);
		
	}
	
	/**
	 * Set values for input nodes
	 * @param values
	 */
	public void setInputValues(List<Double> values) {
		for(int i=0;i<inputs.size();i++)
			inputs.get(i).setValue(values.get(i));
	}

	/**
	 * Binary output based on threshold value of 0.5
	 * 
	 * @param d
	 * @return
	 */
    public static int thresholdFunction(double d) {
        if (d < 0.5)
            return 0;
        else
            return 1;
    }
	
}
