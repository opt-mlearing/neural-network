package edu.packt.neuralnet.customerprofile;

import java.io.IOException;

import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;
import edu.packt.neuralnet.som.Kohonen.KohonenCaseStudyENUM;
import edu.packt.neuralnet.util.Data;
import edu.packt.neuralnet.util.Data.NormalizationTypesENUM;

public class FinancialCompany {

	public static void main(String args[]){
		
		Data cardDataInput  = new Data("data", "financial_company_training.csv");
		
		Data cardDataInputTestRNA   = new Data("data", "financial_company_test.csv");
		
		NormalizationTypesENUM NORMALIZATION_TYPE = Data.NormalizationTypesENUM.MAX_MIN;
		
		try {
			double[][] matrixInput = cardDataInput.rawData2Matrix( cardDataInput );
			
			double[][] matrixInputTestRNA = cardDataInput.rawData2Matrix( cardDataInputTestRNA );
			
			double[][] matrixInputNorm = cardDataInput.normalize(matrixInput, NORMALIZATION_TYPE);
			
			double[][] matrixInputTestRNANorm = cardDataInput.normalize(matrixInputTestRNA, NORMALIZATION_TYPE);
			
			NeuralNet n1 = new NeuralNet();
			n1 = n1.initNet(7, 0, 0, 3);
			
			n1.setTrainSet( matrixInputNorm );
			
			n1.setValidationSet( matrixInputTestRNANorm );
			
			n1.setMaxEpochs( 1000 );
			n1.setLearningRate( 0.1 );
			n1.setTrainType( TrainingTypesENUM.KOHONEN );
			n1.setKohonenCaseStudy( KohonenCaseStudyENUM.FINANCIAL_COMPANY );
			
			NeuralNet n1Trained = new NeuralNet();
			
			n1Trained = n1.trainNet(n1);
			
			System.out.println();
			System.out.println("---------KOHONEN TEST---------");

			double[][] matrixEstimated = n1Trained.netValidation(n1Trained);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
}
