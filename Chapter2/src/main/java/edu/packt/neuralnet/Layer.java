package edu.packt.neuralnet;

import java.util.ArrayList;

public abstract class Layer {

	private ArrayList<Neuron> listOfNeurons;
	protected int numberOfNeuronsInLayer;
	
	public void printLayer(){
	}

	public ArrayList<Neuron> getListOfNeurons() {
		return listOfNeurons;
	}

	public void setListOfNeurons(ArrayList<Neuron> listOfNeurons) {
		this.listOfNeurons = listOfNeurons;
	}

	public int getNumberOfNeuronsInLayer() {
		return numberOfNeuronsInLayer;
	}

	//NEW
	public void setNumberOfNeuronsInLayer(int numberOfNeuronsInLayer) {
		this.numberOfNeuronsInLayer = numberOfNeuronsInLayer;
	}
	
	
	
	
}
