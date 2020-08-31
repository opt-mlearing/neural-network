package edu.packt.neuralnet.learn;

import java.util.ArrayList;

import edu.packt.neuralnet.InputLayer;
import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.Neuron;

public abstract class Training {

    private int epochs;
    private double error;
    private double mse;

    public enum TrainingTypesENUM {
        PERCEPTRON,           // 感知机模型
        ADALINE,              // 自适应模型
        BACKPROPAGATION,      // BP反向传播模型
        LEVENBERG_MARQUARDT;  // L-M快速梯度下降模型.
    }

    /**
     * train基础训练
     * 主要针对单层隐藏层的网络结构.
     *
     * @param neuralNet
     * @return
     */
    public NeuralNet train(NeuralNet neuralNet) {

        ArrayList<Double> inputWeightIn = new ArrayList<Double>();

        // 训练数据的样例大小
        int rows = neuralNet.getTrainSet().length;
        // 训练数据的输入向量长度
        int cols = neuralNet.getTrainSet()[0].length;

        while (this.getEpochs() < neuralNet.getMaxEpochs()) {
            double estimatedOutput = 0.0;
            double realOutput = 0.0;
            for (int i = 0; i < rows; i++) {
                double netValue = 0.0;
                for (int j = 0; j < cols; j++) {
                    inputWeightIn = neuralNet.getInputLayer().getListOfNeurons().get(j).getListOfWeightIn();
                    double inputWeight = inputWeightIn.get(0);
                    netValue = netValue + inputWeight * neuralNet.getTrainSet()[i][j];
                }
                // netValue为全部 神经树突*权值 集合的结果，准备作用激活函数，获得网络输出结果.
                estimatedOutput = this.activationFnc(neuralNet.getActivationFnc(), netValue);
                // 实际标准结果.
                realOutput = neuralNet.getRealOutputSet()[i];
                // 每训练一个样本就设置一次误差
                this.setError(realOutput - estimatedOutput);
                // System.out.println("Epoch: "+this.getEpochs()+" / Error: " + this.getError());
                if (Math.abs(this.getError()) > neuralNet.getTargetError()) {
                    // fix weights
                    InputLayer inputLayer = new InputLayer();
                    inputLayer.setListOfNeurons(this.teachNeuronsOfLayer(cols, i, neuralNet, netValue));
                    neuralNet.setInputLayer(inputLayer);
                }

            }

            this.setMse(Math.pow(realOutput - estimatedOutput, 2.0));
            neuralNet.getListOfMSE().add(this.getMse());
            // 迭代次数自增1.
            this.setEpochs(this.getEpochs() + 1);
        }

        neuralNet.setTrainingError(this.getError());

        return neuralNet;
    }

    private ArrayList<Neuron> teachNeuronsOfLayer(int numberOfInputNeurons, int line, NeuralNet n, double netValue) {
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();
        ArrayList<Double> inputWeightsInNew = new ArrayList<Double>();
        ArrayList<Double> inputWeightsInOld = new ArrayList<Double>();

        for (int j = 0; j < numberOfInputNeurons; j++) {
            inputWeightsInOld = n.getInputLayer().getListOfNeurons().get(j)
                    .getListOfWeightIn();
            double inputWeightOld = inputWeightsInOld.get(0);

            inputWeightsInNew.add(this.calcNewWeight(n.getTrainType(),
                    inputWeightOld, n, this.getError(),
                    n.getTrainSet()[line][j], netValue));

            Neuron neuron = new Neuron();
            neuron.setListOfWeightIn(inputWeightsInNew);
            listOfNeurons.add(neuron);
            inputWeightsInNew = new ArrayList<Double>();
        }

        return listOfNeurons;

    }

    private double calcNewWeight(TrainingTypesENUM trainType,
                                 double inputWeightOld, NeuralNet n, double error,
                                 double trainSample, double netValue) {
        switch (trainType) {
            case PERCEPTRON:
                return inputWeightOld + n.getLearningRate() * error * trainSample;
            case ADALINE:
                return inputWeightOld + n.getLearningRate() * error * trainSample
                        * derivativeActivationFnc(n.getActivationFnc(), netValue);
            default:
                throw new IllegalArgumentException(trainType
                        + " does not exist in TrainingTypesENUM");
        }
    }

    public enum ActivationFncENUM {
        STEP, LINEAR, SIGLOG, HYPERTAN;
    }

    /**
     * 激活函数.
     *
     * @param fnc   使用的激活函数的枚举类型.
     * @param value 数突加权聚合
     * @return value 轴突输出.
     */
    protected double activationFnc(ActivationFncENUM fnc, double value) {
        switch (fnc) {
            case STEP:
                return fncStep(value);
            case LINEAR:
                return fncLinear(value);
            case SIGLOG:
                return fncSigLog(value);
            case HYPERTAN:
                return fncHyperTan(value);
            default:
                throw new IllegalArgumentException(fnc + " does not exist in ActivationFncENUM");
        }
    }

    /**
     * 反向传播梯度.
     *
     * @param fnc   使用的激活函数的枚举类型.
     * @param value
     * @return
     */
    public double derivativeActivationFnc(ActivationFncENUM fnc, double value) {
        switch (fnc) {
            case LINEAR:
                return derivativeFncLinear(value);
            case SIGLOG:
                return derivativeFncSigLog(value);
            case HYPERTAN:
                return derivativeFncHyperTan(value);
            default:
                throw new IllegalArgumentException(fnc + " does not exist in ActivationFncENUM");
        }
    }

    // 阶跃激活函数.
    private double fncStep(double v) {
        if (v >= 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    // 线性激活函数.
    private double fncLinear(double v) {
        return v;
    }

    // sigmoid激活函数.
    private double fncSigLog(double v) {
        return (1.0 / (1.0 + Math.exp(-v)));
    }

    // 正切激活函数
    private double fncHyperTan(double v) {
        return Math.tanh(v);
    }

    // 线性求导
    private double derivativeFncLinear(double v) {
        return 1.0;
    }

    // sigmoid求导.
    private double derivativeFncSigLog(double v) {
        return (v * (1.0 - v));
    }

    // 正切求导.
    private double derivativeFncHyperTan(double v) {
        return (1.0 / Math.pow(Math.cosh(v), 2.0));
    }

    public void printTrainedNetResult(NeuralNet trainedNet) {

        int rows = trainedNet.getTrainSet().length;
        int cols = trainedNet.getTrainSet()[0].length;

        ArrayList<Double> inputWeightIn = new ArrayList<Double>();

        for (int i = 0; i < rows; i++) {
            double netValue = 0.0;
            for (int j = 0; j < cols; j++) {
                inputWeightIn = trainedNet.getInputLayer().getListOfNeurons().get(j).getListOfWeightIn();
                double inputWeight = inputWeightIn.get(0);
                netValue = netValue + inputWeight * trainedNet.getTrainSet()[i][j];

                System.out.print(trainedNet.getTrainSet()[i][j] + "\t");
            }

            double estimatedOutput = this.activationFnc(trainedNet.getActivationFnc(), netValue);

            int colsOutput = trainedNet.getRealMatrixOutputSet()[0].length;

            double realOutput = 0.0;
            for (int k = 0; k < colsOutput; k++) {
                realOutput = realOutput + trainedNet.getRealMatrixOutputSet()[i][k];
            }

            System.out.print(" NET OUTPUT: " + estimatedOutput + "\t");
            System.out.print(" REAL OUTPUT: " + realOutput + "\t");
            double error = estimatedOutput - realOutput;
            System.out.print(" ERROR: " + error + "\n");

        }

    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getMse() {
        return mse;
    }

    public void setMse(double mse) {
        this.mse = mse;
    }

}
