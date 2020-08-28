package edu.packt.neuralnet;

import edu.packt.neuralnet.learn.Training.ActivationFncENUM;
import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;

public class NeuralNetTest {

	public static void main(String[] args) {
		NeuralNetTest test = new NeuralNetTest();
		
		//test.testBackpropagation();
                
                test.testLMA();
	}
	
	private void testBackpropagation(){
		NeuralNet testNet = new NeuralNet();
		
		testNet = testNet.initNet(2, 1, 3, 2);
		
		System.out.println("---------BACKPROPAGATION INIT NET---------");
		
		testNet.printNet(testNet);
		
		NeuralNet trainedNet = new NeuralNet();
		
		// first column has BIAS
		testNet.setTrainSet(new double[][] { { 1.0, 1.0, 0.73 }, { 1.0, 1.0, 0.81 }, { 1.0, 1.0, 0.86 }, 
											 { 1.0, 1.0, 0.95 }, { 1.0, 0.0, 0.45 }, { 1.0, 1.0, 0.70 },
											 { 1.0, 0.0, 0.51 }, { 1.0, 1.0, 0.89 }, { 1.0, 1.0, 0.79 }, { 1.0, 0.0, 0.54 }
									});
		testNet.setRealMatrixOutputSet(new double[][] { {1.0, 0.0}, {1.0, 0.0},	{1.0, 0.0}, 
														{1.0, 0.0},	{1.0, 0.0},	{0.0, 1.0},
														{0.0, 1.0},	{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
									});
		testNet.setMaxEpochs(1000);
		testNet.setTargetError(0.002);
		testNet.setLearningRate(0.1);
		testNet.setTrainType(TrainingTypesENUM.BACKPROPAGATION);
		testNet.setActivationFnc(ActivationFncENUM.SIGLOG);
		testNet.setActivationFncOutputLayer(ActivationFncENUM.LINEAR);
		
		trainedNet = testNet.trainNet(testNet);

		System.out.println();
		System.out.println("---------BACKPROPAGATION TRAINED NET---------");

		testNet.printNet(trainedNet);

	}
        
        private void testLMA(){
		NeuralNet testNet = new NeuralNet();
		
		testNet = testNet.initNet(2, 1, 3, 2);
		
		System.out.println("---------LEVENBERG-MARQUARDT NET---------");
		
		testNet.printNet(testNet);
		
		NeuralNet trainedNet = new NeuralNet();
		
		// first column has BIAS
		testNet.setTrainSet(new double[][] { { 1.0, 1.0, 0.73 }, { 1.0, 1.0, 0.81 }, { 1.0, 1.0, 0.86 }, 
											 { 1.0, 1.0, 0.95 }, { 1.0, 0.0, 0.45 }, { 1.0, 1.0, 0.70 },
											 { 1.0, 0.0, 0.51 }, { 1.0, 1.0, 0.89 }, { 1.0, 1.0, 0.79 }, { 1.0, 0.0, 0.54 }
									});
		testNet.setRealMatrixOutputSet(new double[][] { {1.0, 0.0}, {1.0, 0.0},	{1.0, 0.0}, 
														{1.0, 0.0},	{1.0, 0.0},	{0.0, 1.0},
														{0.0, 1.0},	{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
									});
		testNet.setMaxEpochs(1000);
		testNet.setTargetError(0.002);
		testNet.setLearningRate(0.1);
		testNet.setTrainType(TrainingTypesENUM.LEVENBERG_MARQUARDT);
		testNet.setActivationFnc(ActivationFncENUM.SIGLOG);
		testNet.setActivationFncOutputLayer(ActivationFncENUM.LINEAR);
		
		trainedNet = testNet.trainNet(testNet);

		System.out.println();
		System.out.println("---------BACKPROPAGATION TRAINED NET---------");

		testNet.printNet(trainedNet);

	}        

}
