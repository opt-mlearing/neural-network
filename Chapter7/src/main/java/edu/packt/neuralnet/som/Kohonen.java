package edu.packt.neuralnet.som;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.learn.Training;
import edu.packt.neuralnet.validation.Validation;

public class Kohonen extends Training implements Validation {

	public enum KohonenCaseStudyENUM { CARD, FINANCIAL_COMPANY };
	
	public NeuralNet train(NeuralNet n) {
		
		int rows = n.getTrainSet().length;
		
		n = this.initNet( n );
		
		ArrayList<Double> listOfDistances = new ArrayList<Double>();
		
		double trainData[][] = n.getTrainSet();
		
		for (int epoch = 0; epoch < n.getMaxEpochs(); epoch++) {
			
			//System.out.println("### EPOCH: "+epoch);
		
			for (int row_i = 0; row_i < rows; row_i++) {
				listOfDistances = calcEuclideanDistance(n, trainData, row_i);
				
				int winnerNeuron = listOfDistances.indexOf(Collections.min(listOfDistances));
				
				n = fixWinnerWeights(n, winnerNeuron, row_i);
				
			}
		
		}
		
		return n;
		
	}
	
	private NeuralNet initNet(NeuralNet n) {
		ArrayList<Double> listOfWeightOut = new ArrayList<Double>();
		
		for (int i = 0; i < n.getInputLayer().getNumberOfNeuronsInLayer() * n.getOutputLayer().getNumberOfNeuronsInLayer(); i++) {
			listOfWeightOut.add( 0.0 );
		}
		
		n.getInputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeightOut );
		
		return n;
		
	}
	
	private ArrayList<Double> calcEuclideanDistance(NeuralNet n, double[][] data, int row) {
		ArrayList<Double> listOfDistances = new ArrayList<Double>();
		
		int weight_i = 0;
		for(int cluster_i = 0; cluster_i < n.getOutputLayer().getNumberOfNeuronsInLayer(); cluster_i++) {
			
			double distance = 0.0;
			
			for(int input_j = 0; input_j < n.getInputLayer().getNumberOfNeuronsInLayer(); input_j++) {
				
				double weight = n.getInputLayer().getListOfNeurons().get(0).getListOfWeightOut().get(weight_i);
				
				distance = distance + Math.pow(data[row][input_j] - weight, 2.0);
				
				weight_i++;
				
			}
			
			listOfDistances.add(distance);
			
			//System.out.println("distance normal "+cluster_i+": "+distance);
			
		}
		
		return listOfDistances;
		
	}
	
	private NeuralNet fixWinnerWeights(NeuralNet n, int winnerNeuron, int trainSetRow) {
		int start, last;
		
		start = winnerNeuron * n.getInputLayer().getNumberOfNeuronsInLayer();
		
		if(start < 0) {
			start = 0;
		}
		
		last = start + n.getInputLayer().getNumberOfNeuronsInLayer();
		
		List<Double> listOfOldWeights = new ArrayList<Double>();
		listOfOldWeights = n.getInputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut().subList(start, last);
		
		ArrayList<Double> listOfWeights = new ArrayList<Double>();
		listOfWeights = n.getInputLayer().getListOfNeurons().get( 0 ).getListOfWeightOut();
		
		int col_i = 0;
		for (int j = start; j < last; j++) {
			double trainSetValue = n.getTrainSet()[trainSetRow][col_i];
			double newWeight = listOfOldWeights.get(col_i) + 
					n.getLearningRate() * 
					(trainSetValue - listOfOldWeights.get(col_i));
			
			//System.out.println("newWeight: " + newWeight);
			
			listOfWeights.set(j, newWeight);
			col_i++;
		}
		
		n.getInputLayer().getListOfNeurons().get( 0 ).setListOfWeightOut( listOfWeights );
		
		return n;
		
	}

	public double[][] netValidation(NeuralNet n) {
		int rows = n.getValidationSet().length;
		
		double[][] matrixEstimated = new double[rows][1];
		
		ArrayList<Double> listOfDistances = new ArrayList<Double>();
		
		double validationData[][] = n.getValidationSet();
		KohonenCaseStudyENUM caseStudy = n.getKohonenCaseStudy();
		
		for (int row_i = 0; row_i < rows; row_i++) {
			listOfDistances = calcEuclideanDistance(n, validationData, row_i);
			
			int winnerNeuron = listOfDistances.indexOf(Collections.min(listOfDistances));
			
			switch ( caseStudy ) {
				case CARD:
					switch (winnerNeuron) {
						case 0:
							//CARD
							System.out.println("1");
							matrixEstimated[row_i][0] = 1.0;  
							break;
						case 1:
							System.out.println("-1");
							matrixEstimated[row_i][0] = -1.0;
							break;
						default:
							throw new IllegalArgumentException("Error! Without neural clustering...");
					}
				break;
				case FINANCIAL_COMPANY:
					switch (winnerNeuron) {
						case 0:
							System.out.println("1");
							matrixEstimated[row_i][0] = 1.0;  
							break;
						case 1:
							System.out.println("2");
							matrixEstimated[row_i][0] = 2.0;
							break;
						case 2:
							System.out.println("3");
							matrixEstimated[row_i][0] = 3.0;
							break;
						case 3:
							System.out.println("4");
							matrixEstimated[row_i][0] = 4.0;
							break;
						default:
							throw new IllegalArgumentException("Error! Without neural clustering...");
					}
				break;
				default:
					throw new IllegalArgumentException("Error! Case study does not exist...");
			}
			
		}
		
		return matrixEstimated;
		
	}

}
