package edu.packt.neuralnet.customerprofile;

import java.io.IOException;
import java.util.ArrayList;

import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;
import edu.packt.neuralnet.som.Kohonen.KohonenCaseStudyENUM;
import edu.packt.neuralnet.util.Classification;
import edu.packt.neuralnet.util.Data;
import edu.packt.neuralnet.util.Data.NormalizationTypesENUM;

public class Card {

	public static void main(String args[]){
		
		Data cardDataInput  = new Data("data", "card_inputs_training.csv");
		
		Data cardDataInputTestRNA   = new Data("data", "card_inputs_test.csv");
		Data cardDataOutputTestRNA  = new Data("data", "card_output_test.csv");
		
		NormalizationTypesENUM NORMALIZATION_TYPE = Data.NormalizationTypesENUM.MAX_MIN;
		
		try {
			double[][] matrixInput = cardDataInput.rawData2Matrix( cardDataInput );
			
			double[][] matrixInputTestRNA = cardDataInput.rawData2Matrix( cardDataInputTestRNA );
			
			double[][] matrixOutput = cardDataInput.rawData2Matrix( cardDataOutputTestRNA );
			
			
			double[][] matrixInputNorm = cardDataInput.normalize(matrixInput, NORMALIZATION_TYPE);
			
			double[][] matrixInputTestRNANorm = cardDataInput.normalize(matrixInputTestRNA, NORMALIZATION_TYPE);
			
			NeuralNet n1 = new NeuralNet();
			n1 = n1.initNet(10, 0, 0, 2);
			
			n1.setTrainSet( matrixInputNorm );
			
			n1.setValidationSet( matrixInputTestRNANorm );
			n1.setRealMatrixOutputSet( matrixOutput );
			
			n1.setMaxEpochs(100);
			n1.setLearningRate(0.1);
			n1.setTrainType(TrainingTypesENUM.KOHONEN);
			n1.setKohonenCaseStudy( KohonenCaseStudyENUM.CARD );
			
			NeuralNet n1Trained = new NeuralNet();
			
			n1Trained = n1.trainNet( n1 );
			
			System.out.println();
			System.out.println("---------KOHONEN TEST---------");

			ArrayList<double[][]> listOfArraysToJoin = new ArrayList<double[][]>();
			
			double[][] matrixReal = n1Trained.getRealMatrixOutputSet();
			double[][] matrixEstimated = n1Trained.netValidation(n1Trained);
			
			listOfArraysToJoin.add( matrixReal );
			listOfArraysToJoin.add( matrixEstimated );
			
			double[][] matrixOutputsJoined = new Data().joinArrays(listOfArraysToJoin);
			
			//CONFUSION MATRIX
			Classification classif = new Classification();
			
			double[][] confusionMatrix = classif.calculateConfusionMatrix(-1.0, matrixOutputsJoined);
			classif.printConfusionMatrix(confusionMatrix);
			
			//SENSITIVITY
			System.out.println("SENSITIVITY = " + classif.calculateSensitivity(confusionMatrix));
			
			//SPECIFICITY
			System.out.println("SPECIFICITY = " + classif.calculateSpecificity(confusionMatrix));

			//ACCURACY
			System.out.println("ACCURACY    = " + classif.calculateAccuracy(confusionMatrix));
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
}
