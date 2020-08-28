package edu.packt.neuralnet.util;

public class Classification {

    public double[][] calculateConfusionMatrix(double marginError, double[][] matrix) {

        double[][] confusionMatrix = new double[2][2];

        int rows = matrix.length;
        int cols = matrix[0].length;

        int TP = 0, FP = 0, TN = 0, FN = 0;

        for (int rows_i = 0; rows_i < rows; rows_i++) {

            double[] tempArray = new double[2];

            for (int cols_i = 0; cols_i < cols; cols_i++) {

                double value = matrix[rows_i][cols_i];

                if (value <= marginError) {
                    value = -1.0;
                } else {
                    value = 1.0;
                }

                tempArray[cols_i] = value;

            }

            if (tempArray[0] == 1.0 && tempArray[1] == 1.0) {
                TP++;
            } else if (tempArray[0] == -1.0 && tempArray[1] == -1.0) {
                TN++;
            } else if (tempArray[0] == -1.0 && tempArray[1] == 1.0) {
                FP++;
            } else if (tempArray[0] == 1.0 && tempArray[1] == -1.0) {
                FN++;
            }

        }

        confusionMatrix[0][0] = TP;
        confusionMatrix[0][1] = FP;
        confusionMatrix[1][0] = FN;
        confusionMatrix[1][1] = TN;

        return confusionMatrix;

    }

    public void printConfusionMatrix(double[][] matrix) {
        System.out.println("#### CONFUSION MATRIX ####");
        System.out.println(matrix[0][0] + "\t | \t" + matrix[0][1] + "\t |");
        System.out.println(matrix[1][0] + "\t | \t" + matrix[1][1] + "\t |");
    }

    public double calculateSensitivity(double[][] matrix) {
        double TP = matrix[0][0];
        double FN = matrix[1][0];
        return TP / (TP + FN);
    }

    public double calculateSpecificity(double[][] matrix) {
        double TN = matrix[1][1];
        double FP = matrix[0][1];
        return TN / (FP + TN);
    }

    public double calculateAccuracy(double[][] matrix) {
        double TP = matrix[0][0];
        double TN = matrix[1][1];
        double FN = matrix[1][0];
        double FP = matrix[0][1];

        return (TP + TN) / (TP + FN + FP + TN);

    }

    @SuppressWarnings("unused")
    public double[][] convertToOneColumn(double[][] matrix) {

        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] matrixOneColumn = new double[rows][1];

        for (int rows_i = 0; rows_i < rows; rows_i++) {

            if (matrix[rows_i][0] >= 0.5 && matrix[rows_i][1] < 0.5) {
                matrixOneColumn[rows_i][0] = 1.0;
            } else if (matrix[rows_i][0] < 0.5 && matrix[rows_i][1] >= 0.5) {
                matrixOneColumn[rows_i][0] = 0.0;
            }

        }

        return matrixOneColumn;

    }

}
