package com.green.nn;

import com.green.nn.test.NNUtilities;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Neuron {

    public enum Type {INPUT, HIDDEN, OUTPUT}

	private Map<Neuron, Double> weights;
    private List<Neuron> next;
	private double value;
    private double error;
    private Type type;

	public Neuron() {
		weights = new HashMap<>();
        next = new ArrayList<>();
		value = 0;
        error = 0;
	}
	
	public void compute() {
		for(Entry<Neuron, Double> e: weights.entrySet()) {
			value += e.getKey().getValue() * e.getValue();
		}
	}

    private void calculateError(double desired) {
        error = value * (1.0-value) * (desired - value);
    }

    private void calculateError(Iterable<Neuron> outputs) {
        double outputFactor = 0;
        for(Neuron n: outputs) {
            outputFactor += n.weights.get(this) * n.getError();
        }

        error = value * (1-value) * outputFactor;
    }

    public void adjustWeights(double desired) {
        calculateError(desired);
        adjustWeights();
    }

    public void adjustWeights(Iterable<Neuron> outputs) {
        calculateError(outputs);
        adjustWeights();
    }

    private void adjustWeights() {
        for(Neuron n: weights.keySet()) {
            double newWeight = weights.get(n) + (error * n.getValue());
            weights.put(n, newWeight);
        }
    }

    public void addNext(Neuron n) {
        next.add(n);
    }

    public void addNexts(Iterable<Neuron> neurons) {
        for(Neuron n: neurons)
            addNext(n);
    }

    public void addAxon(Neuron n, double weight) {
        weights.put(n, weight);
        //System.out.println(weight);
    }

	public void addAxon(Neuron n) {
		addAxon(n, NNUtilities.randomDouble());
	}
	
	public void addAxons(Iterable<Neuron> neurons) {
		for(Neuron n: neurons)
			addAxon(n);
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

    public double getError() {
        return error;
    }

    public void setError(double aError) {
        error = aError;
    }
}
