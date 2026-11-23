import java.io.*;
import java.util.*;

public class Classifier extends Matrix {
	public static void main(String[] args) throws IOException {
		// Getting xdata and ydata arrays
		BufferedReader br = new BufferedReader(new FileReader("xdata.txt"));
		BufferedReader br2 = new BufferedReader(new FileReader("ydata.txt"));
		BufferedReader br3 = new BufferedReader(new FileReader("w1.txt"));
		BufferedReader br4 = new BufferedReader(new FileReader("w2.txt"));

		String line;

		ArrayList<double[]> tempList1 = new ArrayList<>();
		ArrayList<double[]> tempList2 = new ArrayList<>();
		ArrayList<double[]> tempList3 = new ArrayList<>();

		while ((line = br.readLine()) != null) {
			String[] strArr = line.split(",");
			double[] doubleArr = new double[strArr.length];
			for (int i = 0; i < strArr.length; i++) {
				doubleArr[i] = Double.parseDouble(strArr[i]);
			}
			tempList1.add(doubleArr);

		}
		// weights1
		while ((line = br3.readLine()) != null) {
			String[] strArr = line.split(",");
			double[] doubleArr = new double[strArr.length];
			for (int i = 0; i < strArr.length; i++) {
				doubleArr[i] = Double.parseDouble(strArr[i]);
			}
			tempList2.add(doubleArr);
		}

		while ((line = br4.readLine()) != null) {
			String[] strArr = line.split(",");
			double[] doubleArr = new double[strArr.length];
			for (int i = 0; i < strArr.length; i++) {
				doubleArr[i] = Double.parseDouble(strArr[i]);
			}
			tempList3.add(doubleArr);
		}

		double[][] xdata = new double[tempList1.size()][tempList1.get(0).length + 1];

		double[][] w1 = new double[tempList2.size()][tempList2.get(0).length];
		double[][] w2 = new double[tempList3.size()][tempList3.get(0).length];
		double[] ydata = new double[xdata.length];
		int temp = 0;
		while ((line = br2.readLine()) != null) {

			ydata[temp] = Double.parseDouble(line);
			// System.out.println(ydata[temp]);
			temp++;
		}
		for (int i = 1; i < tempList1.size(); i++) {
			for (int j = 0; j < tempList1.get(i).length; j++) {
				xdata[i][j] = tempList1.get(i)[j];

			}
		}

		for (int i = 0; i < tempList2.size(); i++) {
			for (int j = 0; j < tempList2.get(i).length; j++) {
				w1[i][j] = tempList2.get(i)[j];

			}
		}
		for (int i = 0; i < tempList3.size(); i++) {
			for (int j = 0; j < tempList3.get(i).length; j++) {
				w2[i][j] = tempList3.get(i)[j];

			}
		}
		// add bias
		for (int i = 0; i < w1.length; i++) {
			xdata[i][0] = 1;

		}
		double[][] Y = new double[ydata.length][10];
		// one-hot encoding Y
		for (int i = 0; i < Y.length; i++) {
			for (int j = 0; j < Y[i].length; j++) {
				if (j + 1 == (int) ydata[i]) {
					Y[i][j] = 1;
				} else {
					Y[i][j] = 0;
				}
			}
		}
		// creating no bias weight arrays
		double[][] w1NoBias = new double[w1.length][w1[0].length];

		// Starting the fun part once again
		int iterator = 500;
		int epoch = 1;
		double learningRate = 0.5;
		double lambda =2.0;

		double[][] h1; // 10000x30
		double[][] yhat = new double[xdata.length][w2.length]; // 10000x10
		int xdataLength = xdata.length;
		System.out.println("*************************************************");
		System.out.println("Name:		Anthony Pearson");
		System.out.println("Date:		4-24-2024");
		System.out.println("*************************************************");
		System.out.println("");
		System.out.println("Training Phase: " + "xdata.txt");
		System.out.println("	=> Number of Entries (n):		" + xdata.length);
		System.out.println("	=> Number of Features (n):		" + "784");

		System.out.println("");
		System.out.println("Starting Gradient Descent: ");
		System.out.println("-------------------------------------------------");

		double loss = 0;
		while (iterator > 0) {
			

			double oldLoss = loss;

			for (int i = 0; i < w1.length; i++) {
				w1NoBias[i][0] = 0;
			}
			for (int i = 0; i < w1.length; i++) {
				for (int j = 1; j < w1[i].length; j++) {
					w1NoBias[i][j] = w1[i][j];
				}
			}
			double[][] w2NoBias = new double[w2.length][w2[0].length];
			for (int i = 0; i < w2.length; i++) {
				w2NoBias[i][0] = 0;
			}
			for (int i = 0; i < w2.length; i++) {
				for (int j = 1; j < w2[i].length; j++) {
					w2NoBias[i][j] = w2[i][j];
				}
			}
		
			double[][] w1T = transpose(w1);
			double[][] w2T = transpose(w2);
			double[][] w1NoBiasT = transpose(w1NoBias);
			double[][] w2NoBiasT = transpose(w2NoBias);
			h1 = activate(multiply(xdata, w1T));

			double[][] h1temp = new double[h1.length][h1[0].length + 1];

			for (int i = 0; i < h1.length; i++) {
				for (int j = 1; j < h1[i].length; j++) {
					h1temp[i][j] = h1[i][j];
				}
				h1temp[i][0] = 1;
			}
			h1 = h1temp;

			yhat = activate(multiply(h1, w2T));

			loss = lossFunction(yhat, Y, w1, w2, lambda, xdataLength);

			double[][] d2 = delta2(yhat, Y);
			double[][] d1 = delta1(d2, w2NoBias, w1NoBiasT, xdata);
			double[][] dw2 = dw2(d2, xdataLength, h1, w2NoBias, lambda);
			double[][] dw1 = dw1(d1, xdataLength, xdata, w1NoBias, lambda);

			for (int i = 0; i < w2.length; i++) {
				for (int j = 0; j < w2[i].length; j++) {
					w2[i][j] = w2[i][j] - (learningRate * dw2[i][j]);

				}

			}
			for (int i = 0; i < w1.length; i++) {
				for (int j = 0; j < w1[i].length; j++) {
					w1[i][j] = w1[i][j] - (learningRate * dw1[i][j]);
				}
			}

			System.out.printf("%s%d\t%s%f\t%s%f%s\t%s%.2f\n", "Epoch: ", epoch, "Loss of ", loss, "Delta = ",
					((loss - oldLoss) / oldLoss) * 100, "%", "Epsilon: ", learningRate);

			iterator -= 1;
			epoch++;

			// System.out.println("Epoch: " + epoch);

		}
		System.out.println("Epochs Required	" + --epoch);
		System.out.println("");
		System.out.println("Testing Phase (first 100 records):");
		System.out.println("-------------------------------------------------");

		double[][] finalYHat = new double[yhat.length][yhat[0].length];
		h1 = activate(multiply(xdata, transpose(w1)));
		double[][] h1temp = new double[h1.length][h1[0].length + 1];

		for (int i = 0; i < h1.length; i++) {
			for (int j = 1; j < h1[i].length; j++) {
				h1temp[i][j] = h1[i][j];
			}
			h1temp[i][0] = 1;
		}
		h1 = h1temp;
		finalYHat = activate(multiply(h1,transpose(w2)));


		double correct = 0;
		for (int i = 0; i < finalYHat.length; i++) {
			double max = Double.MIN_VALUE;
			int prediction = 0;
			for (int j = 0; j < finalYHat[i].length; j++) {
				if (finalYHat[i][j] > max) {
					max = finalYHat[i][j];
					prediction = j+1;
				}
			}
			if (i < 100) {
				System.out.print(
						"Test Record " + (i + 1) + ":	" + (int) ydata[i] + "	Prediction:	" + prediction + " ");
				if (((int) ydata[i]) == prediction) {
					System.out.println("	Correct:  TRUE");
					correct++;

				} else {
					System.out.println("	Correct:  False");
				}

			} else {
				if (((int) ydata[i]) == prediction) {
					correct++;
				}
			}

		}
		System.out.println("");
		System.out.println("	=> Number of Test Entries (n):		" + "100");
		System.out.println("	=> Accuracy:				" + (correct / xdata.length) * 100 + "%");
	}

	// methods
	public static double[][] activate(double[][] X) {
		double[][] result = new double[X.length][X[0].length];
		for (int i = 0; i < X.length; i++) {
			for (int j = 0; j < X[i].length; j++) {
				result[i][j] = 1 / (1 + Math.exp(-X[i][j]));
			}
		}
		return result;
	}

	public static double[][] derivActivate(double[][] X) {
		double[][] temp = activate(X);
		double[][] result = new double[temp.length][temp[0].length];
		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[i].length; j++) {
				result[i][j] = temp[i][j] * (1 - temp[i][j]);

			}
		}

		return result;
	}

	public static double lossFunction(double[][] yhat, double[][] Y, double[][] w1, double[][] w2, double lambda, int xdataLength) {
		double loss = 0;
		for (int i = 0; i < Y.length; i++) {
			for (int j = 0; j < Y[i].length; j++) {
				loss += -Y[i][j] * Math.log(yhat[i][j]) - (1 - Y[i][j]) * Math.log(1 - yhat[i][j]);

			}
		}
		double sumW1Squared = 0;
		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[i].length; j++) {
				sumW1Squared += Math.pow(w1[i][j], 2);
			}
		}
		double sumW2Squared = 0;
		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[i].length; j++) {
				sumW2Squared += Math.pow(w2[i][j], 2);
			}
		}
		double sumWeights = sumW1Squared + sumW2Squared;
		double L2 = lambda / (2 * xdataLength);

		double result = L2 * sumWeights;
		loss = loss + result;

		return loss / xdataLength;
	}

	public static double[][] delta2(double[][] yhat, double[][] Y) {
		double[][] delta2 = new double[Y.length][Y[0].length];
		
		for (int i = 0; i < Y.length; i++) {
			for (int j = 0; j < Y[i].length; j++) {

				delta2[i][j] = yhat[i][j] - Y[i][j];
				// System.out.print(delta2[i][j] + " ");
			}
			// System.out.println("");
		}

		return delta2;
	}

	public static double[][] delta1(double[][] delta2, double[][] w2NoBias, double[][] w1NoBiasT, double[][] xdata) {
		
		double[][] p1 = (multiply(delta2, dropFirstColumn(w2NoBias)));

		double[][] p2 = derivActivate(multiply(xdata, w1NoBiasT));

		double[][] result = new double[p1.length][p1[0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = p1[i][j] * p2[i][j];
			}
		}
		return result;
	}

	public static double[][] dw2(double[][] delta2, int xdataLength, double[][] h1, double[][] w2, double lambda) {
		double result[][] = multiply(transpose(delta2), h1);
		double L2 = lambda/xdataLength;
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				if (result[i][j] == 1) {
					result[i][j] = (result[i][j] + (0 * w2[i][j])) / xdataLength;

				} else {
					result[i][j] = (result[i][j] + (L2 * w2[i][j])) / xdataLength;
				}

			}
		}

		// System.out.println(L2);

		return result;
	}

	public static double[][] dw1(double[][] delta1, int xdataLength, double[][] xdata, double[][] w1, double lambda) {
		double[][] result = multiply(transpose(delta1), xdata);

		double L2 = lambda/xdataLength;

		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				if (result[i][j] == 1) {
					result[i][j] = (result[i][j] + (0 * w1[i][j])) / xdataLength;

				} else {
					result[i][j] = (result[i][j] + (L2 * w1[i][j])) / xdataLength;
				

				}

			}
		}

		return result;
	}

}
