package edu.packt.neuralnet;

import edu.packt.neuralnet.learn.Training.TrainingTypesENUM;

public class NeuralNetTest {

    public static void main(String[] args) {
        NeuralNetTest test = new NeuralNetTest();

        test.testART();
    }

    private void testART() {
        NeuralNet testNet = new NeuralNet();

        //2 inputs because "bias"
        testNet = testNet.initNet(2, 0, 0, 2);

        NeuralNet trainedNet = new NeuralNet();

        //testNet.setTrainSet(new double[][] { { 1.0, 0.0, 1.0 } });

        testNet.setTrainSet(new double[][]{{1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0}, {-1.0, -1.0, 1.0},
                {1.0, 1.0, -1.0}, {-1.0, 1.0, 1.0}, {1.0, -1.0, -1.0}
        });

        //viper and monkey, respectively:
        testNet.setValidationSet(new double[][]{{-1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}});

        testNet.setMaxEpochs(10);
        testNet.setMatchRate(0.5);
        testNet.setTrainType(TrainingTypesENUM.ART);

        trainedNet = testNet.trainNet(testNet);

        System.out.println();
        System.out.println("---------ART VALIDATION NET---------");

        testNet.netValidation(trainedNet);

    }

}
