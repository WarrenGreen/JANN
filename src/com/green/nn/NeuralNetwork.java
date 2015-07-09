package com.green.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
	List<Neuron> inputs;
	public List<Neuron> hidden;
	public List<Neuron> outputs;
	
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

    public static int thresholdFunction(double d) {
        if (d < .5)
            return 0;
        else
            return 1;
    }
	
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(2,3,2);
		nn.setInputValues(Arrays.asList(.3, .8));

        for(int i=0;i<100;i++) {
            nn.setInputValues(Arrays.asList(.3, .7));

            List<Double> outputs = nn.compute();

            if(thresholdFunction(outputs.get(0)) == 1) {
                nn.outputs.get(0).adjustWeights(0);
                for (Neuron n : nn.hidden)
                    n.adjustWeights(nn.outputs);
                System.out.println("HIT+++++++++++++++++++");
            } else if(thresholdFunction(outputs.get(1)) == 0) {
                nn.outputs.get(1).adjustWeights(1);
                for (Neuron n : nn.hidden)
                    n.adjustWeights(nn.outputs);
                System.out.println("HIT+++++++++++++++++++");
            }else{
                System.out.println("Good ==========");
            }
            //nn.outputs.get(1).adjustWeights(1);



            System.out.println();
        }
    }
}
