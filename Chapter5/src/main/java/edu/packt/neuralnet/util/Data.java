package edu.packt.neuralnet.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class Data {

	private String path;
	private String fileName;
	public enum NormalizationTypesENUM {
		MAX_MIN, MAX_MIN_EQUALIZED; 
	}
	
	public Data(String path, String fileName){
		this.path = path;
		this.fileName = fileName;
	}
	
	public Data(){
		
	}
	
	public double[][] joinArrays(ArrayList<double[][]> listOfArraysToJoin){
		
		int rows = listOfArraysToJoin.get(0).length;
		int cols = listOfArraysToJoin.size();
		
		double[][] matrix = new double[rows][cols];
		
		for (int cols_i = 0; cols_i < cols; cols_i++) {
			
			double[][] a = listOfArraysToJoin.get(cols_i);
			
			for (int rows_i = 0; rows_i < rows; rows_i++) {
				
				matrix[rows_i][cols_i] = a[rows_i][0];
				
			}
			
		}
		
		return matrix;
		
	}
	
	public double[][] normalize(double[][] rawMatrix, NormalizationTypesENUM normType) {
		
		int rows = rawMatrix.length;
		int cols = rawMatrix[0].length;
		
		double[][] matrixNorm = new double[rows][cols];
		
		for (int cols_i = 0; cols_i < cols; cols_i++) {
			
			ArrayList<Double> listColumn = new ArrayList<Double>();
			
			for (int rows_j = 0; rows_j < rows; rows_j++) {
				listColumn.add( rawMatrix[rows_j][cols_i] );
			}
			
			double minColValue = Collections.min(listColumn);
			double maxColValue = Collections.max(listColumn);
			
			
			for (int rows_j = 0; rows_j < rows; rows_j++) {
				/*	FONTE: http://equipe.nce.ufrj.br/thome/p_grad/nn_ic/transp/T5a_mlp_detalhes.pdf	*/
				switch (normType) {
				/*	utiliza os valores máximo / mínimo para normalizar 
					linearmente os dados entre [-1,1) ou (-1, 1].	*/
				case MAX_MIN:
					matrixNorm[rows_j][cols_i] = rawMatrix[rows_j][cols_i] / Math.abs( maxColValue );
					break;
				/*	Utiliza os valores máximo e mínimo para normalizar linearmente os
					dados entre [0, 1].		*/
				case MAX_MIN_EQUALIZED:
					//first column has BIAS: do not need to normalize
					if(cols_i > 0) {
						matrixNorm[rows_j][cols_i] = (rawMatrix[rows_j][cols_i] - minColValue) / (maxColValue - minColValue);
					}else {
						matrixNorm[rows_j][cols_i] = rawMatrix[rows_j][cols_i];
					}
					break;
				default:
					throw new IllegalArgumentException(normType
							+ " does not exist in NormalizationTypesENUM");
				}
				
			}
			
		}
		
		return matrixNorm;
		
	}
	
	public double[][] denormalize(double[][] rawMatrix, double[][] matrixNorm, NormalizationTypesENUM normType) {
		
		int rows = matrixNorm.length;
		int cols = matrixNorm[0].length;
		
		double[][] matrixDenorm = new double[rows][cols];
		
		for (int cols_i = 0; cols_i < cols; cols_i++) {
			
			ArrayList<Double> listColumn = new ArrayList<Double>();
			
			for (int rows_j = 0; rows_j < rows; rows_j++) {
				listColumn.add( rawMatrix[rows_j][cols_i] );
			}
			
			double minColValue = Collections.min(listColumn);
			double maxColValue = Collections.max(listColumn);
			
			for (int rows_j = 0; rows_j < rows; rows_j++) {
				/*	Source: http://equipe.nce.ufrj.br/thome/p_grad/nn_ic/transp/T5a_mlp_detalhes.pdf	*/
				switch (normType) {
				/*	linear normalization between [-1,1) ou (-1, 1]	*/
				case MAX_MIN:
					matrixDenorm[rows_j][cols_i] = matrixNorm[rows_j][cols_i] * Math.abs( maxColValue );
					break;
				/*	linear normalization between [0, 1]	*/
				case MAX_MIN_EQUALIZED:
					//first column has BIAS: do not need to denormalize
					if(cols_i > 0) {
						matrixDenorm[rows_j][cols_i] = (matrixNorm[rows_j][cols_i] * (maxColValue - minColValue)) + minColValue;
					} else {
						matrixDenorm[rows_j][cols_i] = matrixNorm[rows_j][cols_i];
					}
					break;
				default:
					throw new IllegalArgumentException(normType
							+ " does not exist in NormalizationTypesENUM");
				}
				
			}
			
		}
		
		return matrixDenorm;
		
	}

	public double[][] rawData2Matrix(Data r) throws IOException {

		String fullPath = defineAbsoluteFilePath( r );

		BufferedReader buffer = new BufferedReader(new FileReader(fullPath));
		
		try {
			StringBuilder builder = new StringBuilder();
			
			String line = buffer.readLine();
			
			int columns = line.split(",").length;
			int rows = 0; 
			while (line != null) {
				builder.append(line);
				builder.append(System.lineSeparator());
				line = buffer.readLine();
				rows++;
			}
			
			double matrix[][] = new double[rows][columns];
			String everything = builder.toString();
			
			Scanner scan = new Scanner( everything );
			rows = 0;
			while(scan.hasNextLine()){
				String[] strVector = scan.nextLine().split(",");
				for (int i = 0; i < strVector.length; i++) {
					matrix[rows][i] = Double.parseDouble(strVector[i]);
				}
				rows++;
			}
			scan.close();
			
			return matrix;

		} finally {
			buffer.close();
		}

	}
	
	private String defineAbsoluteFilePath(Data r) throws IOException {

		String absoluteFilePath = "";

		String workingDir = System.getProperty("user.dir");

		String OS = System.getProperty("os.name").toLowerCase();

		if (OS.indexOf("win") >= 0) {
			absoluteFilePath = workingDir + "\\" + r.getPath() + "\\" + r.getFileName();
		} else {
			absoluteFilePath = workingDir + "/" + r.getPath() + "/" + r.getFileName();
		}

		File file = new File(absoluteFilePath);

		if (file.exists()) {
			System.out.println("File found!");
			System.out.println(absoluteFilePath);
		} else {
			System.err.println("File did not find...");
		}

		return absoluteFilePath;

	}

	public String getPath() {
		return path;
	}

	public void setPath(String path) {
		this.path = path;
	}

	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fileName) {
		this.fileName = fileName;
	}

}
