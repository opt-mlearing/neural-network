package edu.packt.neuralnet.som;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.learn.Training;
import edu.packt.neuralnet.validation.Validation;

public class ART extends Training implements Validation {

	private int SIZE_OF_INPUT_LAYER;
	private int SIZE_OF_OUTPUT_LAYER;
	
	public NeuralNet train(NeuralNet n){
		
		this.initGlobalVars( n );
		
		n = this.initNet( n );
		
		int rows = n.getTrainSet().length;
		
		double[][] trainPatterns = n.getTrainSet();
		
		for (int epoch = 0; epoch < n.getMaxEpochs(); epoch++) {
			
			for (int row_i = 0; row_i < rows; row_i++) {
			
				int winnerNeuron = this.calcWinnerNeuron( n, row_i, trainPatterns );
				
				n = this.setNetOutput( n, winnerNeuron );
				
				boolean isMatched = this.vigilanceTest( n, row_i );
				
				if ( isMatched ) {
					n = this.fixWeights(n, row_i, winnerNeuron);
				}
				
			}
			
		}

		return n;
		
	}
	
	private NeuralNet fixWeights(NeuralNet n, int row_i, int winnerNeuron) {
		ArrayList<Double> listOfWeightIn  = n.getInputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut();
		ArrayList<Double> listOfWeightOut = n.getOutputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut();

		//output layer
		int firstIndex = winnerNeuron * (listOfWeightOut.size() / SIZE_OF_OUTPUT_LAYER);
		int lastIndex  = firstIndex + (listOfWeightOut.size() / SIZE_OF_OUTPUT_LAYER);
		double weightValue = 0.0;
		int input_col = 0;
		for (int i = firstIndex; i < lastIndex; i++) {
			weightValue = listOfWeightIn.get( i ) * n.getTrainSet()[row_i][input_col];
			listOfWeightIn.set(i, weightValue);
			input_col++;
		}
		
		//input layer	
		for (int input_i = firstIndex; input_i < lastIndex ; input_i++) {
			weightValue = (1.0/2.0);
			for (int input_j = 0; input_j < SIZE_OF_INPUT_LAYER ; input_j++) {
				weightValue = weightValue + (listOfWeightIn.get(input_j) * n.getTrainSet()[row_i][input_j]);
			}
			weightValue = listOfWeightIn.get( input_i ) / weightValue;
			listOfWeightOut.set(input_i, weightValue);
		}
		
		n.getInputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeightIn );
		n.getOutputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeightOut );
		
		return n;
			
	}

	private boolean vigilanceTest(NeuralNet n, int row_i) {
		double v1 = 0.0;
		double v2 = 0.0;
		
		for (int i = 0; i < SIZE_OF_INPUT_LAYER; i++) {
			double weightIn  = n.getInputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut().get( i );
			double trainPattern = n.getTrainSet()[row_i][i];
			
			v1 = v1 + (weightIn * trainPattern);
			
			v2 = v2 + (trainPattern * trainPattern);
		}
		
		double vigilanceValue = v1 / v2;
		
		if(vigilanceValue > n.getMatchRate()){
			return true;
		} else {
			return false;
		}
		
	}

	private NeuralNet setNetOutput(NeuralNet n, int winnerNeuron) {
		for (int i = 0; i < SIZE_OF_OUTPUT_LAYER; i++) {
			if( i == winnerNeuron){
				n.getOutputLayer().getListOfNeurons().get( i ).setOutputValue( 1.0 );
			} else {
				n.getOutputLayer().getListOfNeurons().get( i ).setOutputValue( 0.0 );
			}
		}
		
		return n;
		
	}

	private int calcWinnerNeuron(NeuralNet n, int row_i, double[][] patterns) {
		
		ArrayList<Double> listOfEstimatedOutput = new ArrayList<Double>();
		
		int neuron_i = 0;
		
		for (int i = 0; i < SIZE_OF_OUTPUT_LAYER; i++) {
			
			double netValue = 0.0;
			
			for (int j = 0; j < SIZE_OF_INPUT_LAYER; j++) {
				double weightIn = n.getOutputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut().get(neuron_i);
				
				netValue = netValue + (weightIn * patterns[row_i][j]);
				
				neuron_i++;
			}
			
			listOfEstimatedOutput.add(netValue);
			
		}
		
		int winnerNeuron = listOfEstimatedOutput.indexOf( Collections.max(listOfEstimatedOutput) );
		
		return winnerNeuron;
	}

	private NeuralNet initNet(NeuralNet n) {
		ArrayList<Double> listOfWeightIn = new ArrayList<Double>();
		ArrayList<Double> listOfWeightOut = new ArrayList<Double>();
		
		for (int i = 0; i < SIZE_OF_INPUT_LAYER * SIZE_OF_OUTPUT_LAYER; i++) {
			listOfWeightIn.add( 1.0 );
		}
		
		for (int i = 0; i < SIZE_OF_INPUT_LAYER * SIZE_OF_OUTPUT_LAYER; i++) {
			double initValue = 1.0 / ( 1 + SIZE_OF_INPUT_LAYER );
			listOfWeightOut.add( initValue );
		}
		
		n.getInputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeightIn );
		n.getOutputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeightOut );
		
		return n;
	}
	
	private void initGlobalVars(NeuralNet n) {
		SIZE_OF_INPUT_LAYER  = n.getInputLayer().getNumberOfNeuronsInLayer();
		SIZE_OF_OUTPUT_LAYER = n.getOutputLayer().getNumberOfNeuronsInLayer();
	}

	@Override
	public double[][] netValidation(NeuralNet n) {

		this.initGlobalVars( n );
		
		int rows = n.getValidationSet().length;
		
		double[][] matrixEstimated = new double[rows][SIZE_OF_OUTPUT_LAYER];
		
		double[][] validationData = n.getValidationSet();
		
		for (int row_i = 0; row_i < rows; row_i++) {
			
			int winnerNeuron = this.calcWinnerNeuron(n, row_i, validationData);
			
			n = this.setNetOutput( n, winnerNeuron );
			
			for (int j = 0; j < SIZE_OF_OUTPUT_LAYER; j++) {
				matrixEstimated[row_i][j] = n.getOutputLayer().getListOfNeurons().get( j ).getOutputValue();
			}
			
		}
		
		System.out.println("Matrix:");
		System.out.println(Arrays.deepToString(matrixEstimated).toString());
		
		return matrixEstimated;
		
	}
	
	
}
