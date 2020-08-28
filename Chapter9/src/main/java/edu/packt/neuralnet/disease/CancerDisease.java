package edu.packt.neuralnet.disease;

import java.io.IOException;
import java.util.ArrayList;

import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.learn.Training.ActivationFncENUM;
import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;
import edu.packt.neuralnet.util.Chart;
import edu.packt.neuralnet.util.Classification;
import edu.packt.neuralnet.util.Data;
import edu.packt.neuralnet.util.Data.NormalizationTypesENUM;

public class CancerDisease {

	public static void main(String args[]){
		
		Data diseaseDataInput  = new Data("data", "breast_cancer_inputs_training.csv");
		Data diseaseDataOutput = new Data("data", "breast_cancer_output_training.csv");
		
		Data diseaseDataInputTestRNA  = new Data("data", "breast_cancer_inputs_test.csv");
		Data diseaseDataOutputTestRNA = new Data("data", "breast_cancer_output_test.csv");
		
		NormalizationTypesENUM NORMALIZATION_TYPE = Data.NormalizationTypesENUM.MAX_MIN;
		
		try {
			double[][] matrixInput  = diseaseDataInput.rawData2Matrix( diseaseDataInput );
			double[][] matrixOutput = diseaseDataOutput.rawData2Matrix( diseaseDataOutput );
			
			double[][] matrixInputTestRNA  = diseaseDataOutput.rawData2Matrix( diseaseDataInputTestRNA );
			double[][] matrixOutputTestRNA = diseaseDataOutput.rawData2Matrix( diseaseDataOutputTestRNA );
			
			double[][] matrixInputNorm = diseaseDataInput.normalize(matrixInput, NORMALIZATION_TYPE);
			
			double[][] matrixInputTestRNANorm = diseaseDataOutput.normalize(matrixInputTestRNA, NORMALIZATION_TYPE);
			
			NeuralNet n1 = new NeuralNet();
			n1 = n1.initNet(9, 1, 5, 1);
			
			n1.setTrainSet( matrixInputNorm );
			n1.setRealMatrixOutputSet( matrixOutput );
			
			n1.setMaxEpochs(1000);
			n1.setTargetError(0.00001);
			n1.setLearningRate(0.9);
			n1.setTrainType(TrainingTypesENUM.BACKPROPAGATION);
			n1.setActivationFnc(ActivationFncENUM.SIGLOG);
			n1.setActivationFncOutputLayer(ActivationFncENUM.SIGLOG);
			
			NeuralNet n1Trained = new NeuralNet();
			
			n1Trained = n1.trainNet(n1);
			
			System.out.println();

			//ERROR:
			Chart c1 = new Chart();
			c1.plotXYData(n1.getListOfMSE().toArray(), "MSE Error", "Epochs", "MSE Value");
			
			//TEST:
			n1Trained.setTrainSet( matrixInputTestRNANorm );
			n1Trained.setRealMatrixOutputSet( matrixOutputTestRNA );;
			
			double[][] matrixOutputRNATest = n1Trained.getNetOutputValues(n1Trained);
			
			ArrayList<double[][]> listOfArraysToJoinTest = new ArrayList<double[][]>();
			listOfArraysToJoinTest.add( matrixOutputTestRNA );
			listOfArraysToJoinTest.add( matrixOutputRNATest );
			
			double[][] matrixOutputsJoinedTest = new Data().joinArrays(listOfArraysToJoinTest);
			
			Chart c3 = new Chart();
			c3.plotBarChart(matrixOutputsJoinedTest, "Real x Estimated - Test Data", "Breast Cancer Data", "Diagnosis (0: BEN / 1: MAL)");

			
			//CONFUSION MATRIX
			Classification classif = new Classification();
			
			double[][] confusionMatrix = classif.calculateConfusionMatrix(0.6, matrixOutputsJoinedTest);
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
