package edu.packt.neuralnet.util;

import java.io.IOException;

public class DataTest {

    @SuppressWarnings("unused")
    public static void main(String args[]) {

        Data d = new Data();

        d.setPath("data");
        d.setFileName("01_12_2014_Belem_v1.csv");

        try {
            double[][] matrix = d.rawData2Matrix(d);

            double[][] matrixNorm1 = d.normalize(matrix, Data.NormalizationTypesENUM.MAX_MIN);

            double[][] matrixNorm2 = d.normalize(matrix, Data.NormalizationTypesENUM.MAX_MIN_EQUALIZED);

            double[][] matrixDenorm1 = d.denormalize(matrix, matrixNorm1, Data.NormalizationTypesENUM.MAX_MIN);

            double[][] matrixDenorm2 = d.denormalize(matrix, matrixNorm2, Data.NormalizationTypesENUM.MAX_MIN_EQUALIZED);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}


