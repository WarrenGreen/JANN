package com.green.nn;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class Neuron {
	private Map<Neuron, Double> weights;
	private double value;
	
	public Neuron() {
		weights = new HashMap<>();
		value = 0;
	}
	
	public void compute() {
		for(Entry<Neuron, Double> e: weights.entrySet()) {
			value += e.getKey().getValue() * e.getValue();
		}
	}
	
	public void addAxon(Neuron n, double weight) {
		weights.put(n, weight);
	}
	
	public void addAxons(Iterable<Neuron> neurons, double weight) {
		for(Neuron n: neurons)
			addAxon(n, weight);
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}
	
}
