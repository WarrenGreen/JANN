package com.green.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
	List<Neuron> inputs;
	List<Neuron> hidden;
	List<Neuron> outputs;
	
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

        for(Neuron n: hidden) {
            n.addNexts(outputs);
        }
	}
	
	public List<Double> compute() {
		List<Double> outputValues = new ArrayList<>();
		
		for(Neuron n: hidden)
			n.compute();
		
		for(Neuron n: outputs) {
			n.compute();
			outputValues.add(n.getValue());
		}
		
		return outputValues;
	}
	
	public void setInputValues(List<Double> values) {
		for(int i=0;i<inputs.size();i++)
			inputs.get(i).setValue(values.get(i));
	}
	
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(1, 1, 1);
		nn.setInputValues(Arrays.asList(.3));

        for(int i=0;i<10;i++) {
            nn.setInputValues(Arrays.asList(.3));
            for (Double d : nn.compute())
                System.out.println(d);

            nn.outputs.get(0).adjustWeights(0);
            //nn.outputs.get(1).adjustWeights(1);

            for (Neuron n : nn.hidden)
                n.adjustWeights(nn.outputs);

            System.out.println();
        }
    }
}
