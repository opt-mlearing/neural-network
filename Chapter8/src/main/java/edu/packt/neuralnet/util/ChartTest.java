package edu.packt.neuralnet.util;

import java.io.IOException;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.DefaultPieDataset;

public class ChartTest {

	public static void main(String args[]) {

		//new ChartTest().libTest();
		
		Data d = new Data();
		d.setPath("data");
		d.setFileName("01_12_2014_Belem_v1.csv");
		
		try {
			
			double[][] matrix = d.rawData2Matrix( d );
			
			Chart c = new Chart();
			
			c.plotXYData( matrix, "Input Data", "Data", "Value" );
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	@SuppressWarnings("unused")
	private void libTest(){
		// create a dataset...
		DefaultPieDataset data = new DefaultPieDataset();
		data.setValue("Category 1", 43.2);
		data.setValue("Category 2", 27.9);
		data.setValue("Category 3", 79.5);
		// create a chart...
		JFreeChart chart = ChartFactory.createPieChart("Sample Pie Chart",
				data, true, // legend?
				true, // tooltips?
				false // URLs?
				);

		// create and display a frame...
		ChartFrame frame = new ChartFrame("First", chart);
		frame.pack();
		frame.setVisible(true);
	}

}
