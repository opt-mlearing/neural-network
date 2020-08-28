package edu.packt.neuralnet.util;

import java.awt.BasicStroke;
import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class Chart {
	
	public enum ChartPlotTypeENUM {
		FULL_DATA, COMPARISON;
	}

	public void plotXYData(Object[] vector, String chartTitle, String xAxisLabel, String yAxisLabel) {
		
		int length = vector.length;
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		
		XYSeries series = new XYSeries( yAxisLabel );
		
		for (int i = 0; i < length; i++) {
				
	        series.add( (i+1), (double) vector[i] );
			
		}
		
		dataset.addSeries(series);
		
		JFreeChart chart = ChartFactory.createXYLineChart(
			chartTitle,		// chart title
            xAxisLabel,     // x axis label
            yAxisLabel,     // y axis label
            dataset,        // data
            PlotOrientation.VERTICAL,
            true,           // include legend
            true,           // tooltips
            false           // urls
        );
		
        ChartFrame frame = new ChartFrame("Neural Net Chart", chart);
		frame.pack();
		frame.setVisible(true);

	}
	
	public void plotXYData(double[][] matrix, String chartTitle, String xAxisLabel, String yAxisLabel, ChartPlotTypeENUM chartPlotType) {
		
		int rows = matrix.length;
		int cols = matrix[0].length;
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		
		for (int cols_i = 0; cols_i < cols; cols_i++) {

			XYSeries seriesMin = null;
			XYSeries seriesMax = null;
			
			XYSeries series = null;
			
			switch (chartPlotType) {
				case FULL_DATA:
					series = new XYSeries( selectTemperatureSeriesName(cols_i) );
					break;
				case COMPARISON:
					series = new XYSeries( selectComparisonSeriesName(cols_i) );
					break;
				default:
					throw new IllegalArgumentException(chartPlotType + " does not exist in ChartPlotTypeENUM");
			}
			
			if(cols_i > 0) {
				seriesMin = new XYSeries( "-1.0 °C" );
				seriesMax = new XYSeries( "+1.0 °C" );
			}
			
			for (int rows_i = 0; rows_i < rows; rows_i++) {
				
				if(cols_i > 0) {
					seriesMin.add( (rows_i+1), matrix[rows_i][cols_i] - 1.0 );
					seriesMax.add( (rows_i+1), matrix[rows_i][cols_i] + 1.0 );
				}
				
		        series.add( (rows_i+1), matrix[rows_i][cols_i] );
				
			}
			
			dataset.addSeries(series);
			
			if(cols_i > 0) {
				dataset.addSeries(seriesMin);
				dataset.addSeries(seriesMax);
			}
			
			
		}
		
		JFreeChart chart = ChartFactory.createXYLineChart(
			chartTitle,		// chart title
            xAxisLabel,     // x axis label
            yAxisLabel,     // y axis label
            dataset,        // data
            PlotOrientation.VERTICAL,
            true,           // include legend
            true,           // tooltips
            false           // urls
        );
		
		XYPlot plot = chart.getXYPlot();
		
		chart.getXYPlot().getRangeAxis().setRange(-2, 2);
		
		//dashed line (min and max):
		XYLineAndShapeRenderer rendererMin = (XYLineAndShapeRenderer) plot.getRenderer();
		XYLineAndShapeRenderer rendererMax = (XYLineAndShapeRenderer) plot.getRenderer();
		rendererMin.setSeriesPaint(2, Color.black);
		rendererMax.setSeriesPaint(3, Color.black);
		rendererMin.setSeriesStroke(2, new BasicStroke(
                2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                1.0f, new float[] {1.0f, 6.0f}, 0.0f
            ));
		
		rendererMax.setSeriesStroke(3, new BasicStroke(
                2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                1.0f, new float[] {1.0f, 6.0f}, 0.0f
            ));
		plot.setRenderer(0, rendererMin);
		plot.setRenderer(1, rendererMax);
		

		ChartFrame frame = new ChartFrame("Neural Net Chart", chart);
		frame.pack();
		frame.setVisible(true);

	}
	
	private String selectComparisonSeriesName(int index){
		switch (index) {
		case 0:
			return "Real";
		case 1:
			return "Estimated";
		default:
			return "Undefined";
		}
	}
	
	private String selectTemperatureSeriesName(int index){
		switch (index) {
		case 0:
			return "TempMean";
		case 1:
			return "PrecipMean";	
		case 2:
			return "Insolation";
		case 3:
			return "RelHumiMean";
		case 4:
			return "WindSpeedMean";
		default:
			return "Undefined";
		}
	}
	
	public void plotBarChart(double[][] matrix, String chartTitle, String xAxisLabel, String yAxisLabel) {
		
		int rows = matrix.length;
		int cols = matrix[0].length;
		
		DefaultCategoryDataset dataset = new DefaultCategoryDataset( );
		
		for (int rows_i = 0; rows_i < rows; rows_i++) {
		
			for (int cols_i = 0; cols_i < cols; cols_i++) {
				
				/*
				if ( numberOfNeuronsOutputLayer > 1 ) {
					
				}
				*/
				
				double realValue = matrix[rows_i][cols_i];
				
				if (cols_i == 0) {
					
					dataset.addValue( realValue, "Real" , rows_i+"" );
					
				} else {
					
					dataset.addValue( realValue, "Estimated" , rows_i+"" );
					
				}
				
			}
			
		}
		
		JFreeChart chart = ChartFactory.createBarChart(
			chartTitle,		// chart title
            xAxisLabel,     // x axis label
            yAxisLabel,     // y axis label
            dataset,        // data
            PlotOrientation.VERTICAL,
            true,           // include legend
            true,           // tooltips
            false           // urls
        );
		
		ChartFrame frame = new ChartFrame("Neural Net Chart", chart);
		frame.pack();
		frame.setVisible(true);

	}
	
}
