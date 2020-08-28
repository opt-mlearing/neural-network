package edu.packt.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;

public class HiddenLayer extends Layer {

    public ArrayList<HiddenLayer> initLayer(HiddenLayer hiddenLayer,
                                            ArrayList<HiddenLayer> listOfHiddenLayer, InputLayer inputLayer,
                                            OutputLayer outputLayer) {

        ArrayList<Double> listOfWeightIn = new ArrayList<Double>();
        ArrayList<Double> listOfWeightOut = new ArrayList<Double>();
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();

        int numberOfHiddenLayers = listOfHiddenLayer.size();

        for (int hdn_i = 0; hdn_i < numberOfHiddenLayers; hdn_i++) {
            for (int neuron_i = 0; neuron_i < hiddenLayer.getNumberOfNeuronsInLayer(); neuron_i++) {
                Neuron neuron = new Neuron();

                int limitIn = 0;
                int limitOut = 0;

                if (hdn_i == 0) { // first
                    limitIn = inputLayer.getNumberOfNeuronsInLayer();
                    if (numberOfHiddenLayers > 1) {
                        limitOut = listOfHiddenLayer.get(hdn_i + 1).getNumberOfNeuronsInLayer();
                    } else if (numberOfHiddenLayers == 1) {
                        limitOut = outputLayer.getNumberOfNeuronsInLayer();
                    }
                } else if (hdn_i == numberOfHiddenLayers - 1) { // last
                    limitIn = listOfHiddenLayer.get(hdn_i - 1).getNumberOfNeuronsInLayer();
                    limitOut = outputLayer.getNumberOfNeuronsInLayer();
                } else { // middle
                    limitIn = listOfHiddenLayer.get(hdn_i - 1).getNumberOfNeuronsInLayer();
                    limitOut = listOfHiddenLayer.get(hdn_i + 1).getNumberOfNeuronsInLayer();
                }

                limitIn = limitIn - 1;  //bias is not connected
                limitOut = limitOut - 1;  //bias is not connected

                if (neuron_i >= 1) { //bias has no input
                    for (int k = 0; k <= limitIn; k++) {
                        listOfWeightIn.add(neuron.initNeuron());
                        //listOfWeightIn.add(neuron.initNeuron(k, neuron_i, 1));
                    }
                }
                for (int k = 0; k <= limitOut; k++) {
                    listOfWeightOut.add(neuron.initNeuron());
                    //listOfWeightOut.add(neuron.initNeuron(k, neuron_i, 2));
                }

                neuron.setListOfWeightIn(listOfWeightIn);
                neuron.setListOfWeightOut(listOfWeightOut);
                listOfNeurons.add(neuron);

                listOfWeightIn = new ArrayList<Double>();
                listOfWeightOut = new ArrayList<Double>();

            }

            listOfHiddenLayer.get(hdn_i).setListOfNeurons(listOfNeurons);

            listOfNeurons = new ArrayList<Neuron>();

        }

        return listOfHiddenLayer;

    }

    public void printLayer(ArrayList<HiddenLayer> listOfHiddenLayer) {
        if (listOfHiddenLayer.size() > 0) {
            System.out.println("### HIDDEN LAYER ###");
            int h = 1;
            for (HiddenLayer hiddenLayer : listOfHiddenLayer) {
                System.out.println("Hidden Layer #" + h);
                int n = 1;
                for (Neuron neuron : hiddenLayer.getListOfNeurons()) {
                    System.out.println("Neuron #" + n);
                    System.out.println("Input Weights:");
                    System.out.println(Arrays.deepToString(neuron
                            .getListOfWeightIn().toArray()));
                    System.out.println("Output Weights:");
                    System.out.println(Arrays.deepToString(neuron
                            .getListOfWeightOut().toArray()));
                    n++;
                }
                h++;
            }
        }
    }

    public void setNumberOfNeuronsInLayer(int numberOfNeuronsInLayer) {
        this.numberOfNeuronsInLayer = numberOfNeuronsInLayer + 1; //BIAS
    }

}
