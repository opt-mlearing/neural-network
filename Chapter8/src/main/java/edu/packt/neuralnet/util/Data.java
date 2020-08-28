package edu.packt.neuralnet.util;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import javax.imageio.ImageIO;

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
					matrixNorm[rows_j][cols_i] = rawMatrix[rows_j][cols_i] 
							/ ( Math.abs( maxColValue ) == 0.0 ? 1.0 : Math.abs( maxColValue ) );
					break;
				/*	Utiliza os valores máximo e mínimo para normalizar linearmente os
					dados entre [0, 1].		*/
				case MAX_MIN_EQUALIZED:
					//first column has BIAS: do not need to normalize
					if(cols_i > 0) {
						matrixNorm[rows_j][cols_i] = (rawMatrix[rows_j][cols_i] - minColValue) 
								/ ( (maxColValue - minColValue) == 0.0 ? 1.0 : (maxColValue - minColValue) );
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

	public int[] getMainImagePixels( Data r ){
		
		int[] pixelsVector = null;
		
		try {
			String fullPath = defineAbsoluteFilePath( r );
			
			BufferedImage image = ImageIO.read( new File( fullPath ) );
			
			//int width = image.getWidth();	//col
			//int height = image.getHeight();	//row
			
			//System.out.println("### IMAGE ###");
			//System.out.println("Original Width:  " + width);
			//System.out.println("Original Height: " + height);
			
			int ROW_INIT = 24;
			int ROW_LAST = 34;
			int ROW_GAP  = 5;
			
			int COL_INIT = 4;
			int COL_LAST = 44;
			
			int col = 0;
			
			pixelsVector = new int[ 120 ]; //only inputs and bias
			
			for (int row_i = ROW_INIT; row_i <= ROW_LAST; row_i = row_i + ROW_GAP) {
				
				for (int col_j = COL_INIT; col_j < COL_LAST; col_j++) {
					
					if(col == 0) { //BIAS
						pixelsVector[col]  = 1;
					} else {
						pixelsVector[col]  = image.getRGB(row_i, col_j);
					}
					
					col++;
					
	         	}
				
			}
			
			System.out.println("COL: "+col);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return pixelsVector;
		
	}
	
	public void writeOutputDataCSV(ArrayList<Integer> pixelsOutputVectorList, Data dataTraningOutputPath) {
		PrintWriter writer;
		try {
			
			String fullPath = defineAbsoluteFilePath( dataTraningOutputPath );
			
			writer = new PrintWriter(fullPath, "UTF-8");
			
			for (int i = 0; i < pixelsOutputVectorList.size(); i++) {
				
				writer.println( pixelsOutputVectorList.get( i ) );
				
			}
			
			writer.close();
			
		} catch (FileNotFoundException e) {
			System.err.println("Error! File not found!");
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			System.err.println("Error! Unsupported encoding!");
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public void writeDataCSV(ArrayList<int[]> pixelsList, Data dataTraningPath) {
		PrintWriter writer;
		try {
			
			String fullPath = defineAbsoluteFilePath( dataTraningPath );
			
			writer = new PrintWriter(fullPath, "UTF-8");
			
			for (int i = 0; i < pixelsList.size(); i++) {
				
				int[] v = pixelsList.get( i );
				
				for (int j = 0; j < v.length; j++) {
					
					writer.print( v[j] + (j == (v.length - 1) ? "" : ",") );
					
				}
				writer.print("\n");
				
			}
			
			writer.close();
			
		} catch (FileNotFoundException e) {
			System.err.println("Error! File not found!");
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			System.err.println("Error! Unsupported encoding!");
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
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
