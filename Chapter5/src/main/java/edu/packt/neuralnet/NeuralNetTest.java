package edu.packt.neuralnet;

import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;

public class NeuralNetTest {

	public static void main(String[] args) {
		NeuralNetTest test = new NeuralNetTest();
		
		test.testKohonen();
	}
	
	private void testKohonen(){
		NeuralNet testNet = new NeuralNet();
		
		//2 inputs because "bias"
		testNet = testNet.initNet(2, 0, 0, 2);
		
		NeuralNet trainedNet = new NeuralNet();
		
		testNet.setTrainSet(new double[][] { { 1.0, -1.0, 1.0 }, { -1.0, -1.0, -1.0 }, { -1.0, -1.0, 1.0 }, 
											 { 1.0, 1.0, -1.0 }, { -1.0, 1.0, 1.0   }, { 1.0, -1.0, -1.0 }
									});
		
		//viper and monkey, respectively:
		testNet.setValidationSet(new double[][] { {-1.0, 1.0, -1.0}, {1.0, 1.0, 1.0} } );
		
		testNet.setMaxEpochs(10);
		testNet.setLearningRate(0.1);
		testNet.setTrainType(TrainingTypesENUM.KOHONEN);
		
		trainedNet = testNet.trainNet(testNet);

		System.out.println();
		System.out.println("---------KOHONEN VALIDATION NET---------");

		testNet.netValidation(trainedNet);

	}

}
