import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class Neural {
	public static void main(String args[]) {
		if (args.length == 0) {
			System.out.println("Invalid Number of Input Arguments");
			return;
		}
		int flag = Integer.valueOf(args[0]);
		double x1 = 0;
		double x2 = 0;
		if (flag < 600) {
			x1 = Double.valueOf(args[10]);
			x2 = Double.valueOf(args[11]);
		}
		double[] weights = new double[9];
		for (int i = 1; i < 10; i++) {
			weights[i - 1] = Double.valueOf(args[i]);
		}
		if (flag == 100) {
			double[] UandV = UsAndVs(weights, x1, x2);
			for (double d : UandV) {
				System.out.print(String.format("%.5f", d) + " ");
			}
		}
		if (flag == 200) {
			double[] UandV = UsAndVs(weights, x1, x2);
			double y = Double.valueOf(args[12]);
			double error = (.5) * (Math.pow((UandV[5] - y), 2));
			double partialOut = UandV[5] - y;
			double partialInter = partialOut * (UandV[5] * (1 - UandV[5]));
			System.out.print(String.format("%.5f", error) + " ");
			System.out.print(String.format("%.5f", partialOut) + " ");
			System.out.print(String.format("%.5f", partialInter));
		}
		if (flag == 300) {
			double[] UandV = UsAndVs(weights, x1, x2);
			double y = Double.valueOf(args[12]);
			double[] partialD = partialDerivatives(UandV, y, weights);
			for (int i = 2; i < partialD.length; i++) {
				System.out.print(String.format("%.5f", partialD[i]) + " ");
			}
		}
		if (flag == 400) {
			double[] UandV = UsAndVs(weights, x1, x2);
			double y = Double.valueOf(args[12]);
			double[] partialD = partialDerivatives(UandV, y, weights);
			double[] partialDW = partialDerivativesWeights(UandV, partialD, x1, x2);
			for (double weight : partialDW) {
				System.out.print(String.format("%.5f", weight) + " ");
			}
		}
		if (flag == 500) {
			double[] UandV = UsAndVs(weights, x1, x2);
			double y = Double.valueOf(args[12]);
			double[] partialD = partialDerivatives(UandV, y, weights);
			double N = Double.valueOf(args[13]);
			double error = (.5) * (Math.pow((UandV[5] - y), 2));
			double[] partialDW = partialDerivativesWeights(UandV, partialD, x1, x2);
			for (double weight : weights) {
				System.out.print(String.format("%.5f", weight) + " ");
			}
			System.out.println();
			System.out.println(String.format("%.5f", error) + " ");
			double newWeights[] = new double[9];
			newWeights = updateWeights(partialDW, weights, N);
			for (double weight : newWeights) {
				System.out.print(String.format("%.5f", weight) + " ");
			}
			double[] newUandV = UsAndVs(newWeights, x1, x2);
			double newError = (.5) * (Math.pow((newUandV[5] - y), 2));
			System.out.println();
			System.out.println(String.format("%.5f", newError) + " ");
		}
		if (flag == 600) {
			double N = Double.valueOf(args[10]);
			// Gets List of X1s X2s and Y values
			ArrayList<ArrayList<Double>> EvalList = readFile("./hw2_midterm_A_eval.txt");
			ArrayList<ArrayList<Double>> TrainList = readFile("./hw2_midterm_A_train.txt");
			ArrayList<Double> X1s = TrainList.get(0);
			ArrayList<Double> X2s = TrainList.get(1);
			ArrayList<Double> ys = TrainList.get(2);
			ArrayList<Double> EvalX1s = EvalList.get(0);
			ArrayList<Double> EvalX2s = EvalList.get(1);
			ArrayList<Double> Evalys = EvalList.get(2);
			// Initializes other Variables
			ArrayList<Double> setError = new ArrayList<>();
			ArrayList<double[]> weightList = new ArrayList<double[]>();
			// stochastic gradient descent
			for (int i = 0; i < X1s.size(); i++) {
				double[] UandV = UsAndVs(weights, X1s.get(i), X2s.get(i));
				double[] partialD = partialDerivatives(UandV, ys.get(i), weights);
				double[] partialDW = partialDerivativesWeights(UandV, partialD, X1s.get(i), X2s.get(i));
				double[] newWeights = updateWeights(partialDW, weights, N);
				weights = newWeights;
				weightList.add(newWeights);
				double evalSetError = getError(EvalX1s, EvalX2s, Evalys, weights);
				setError.add(evalSetError);
			}
			// Prints everything
			for (int i = 0; i < X1s.size(); i++) {
				System.out.println(String.format("%.5f", X1s.get(i)) + " " + String.format("%.5f", X2s.get(i)) + " "
						+ String.format("%.5f", ys.get(i)));
				for (double weight : weightList.get(i)) {
					System.out.print(String.format("%.5f", weight) + " ");
				}
				System.out.println();
				System.out.println(String.format("%.5f", setError.get(i)));
			}
		}
		if (flag == 700) {
			double N = Double.valueOf(args[10]);
			double T = Double.valueOf(args[11]);
			// Gets List of X1s X2s and Y values
			ArrayList<ArrayList<Double>> EvalList = readFile("./hw2_midterm_A_eval.txt");
			ArrayList<ArrayList<Double>> TrainList = readFile("./hw2_midterm_A_train.txt");
			ArrayList<Double> X1s = TrainList.get(0);
			ArrayList<Double> X2s = TrainList.get(1);
			ArrayList<Double> ys = TrainList.get(2);
			ArrayList<Double> EvalX1s = EvalList.get(0);
			ArrayList<Double> EvalX2s = EvalList.get(1);
			ArrayList<Double> Evalys = EvalList.get(2);
			// Initializes other Variables
			ArrayList<double[]> weightList = new ArrayList<double[]>();
			weightList.add(weights);
			int weightLoc = 0;
			for (int t = 0; t < T; t++) {
				// Stochastic Gradient Descent
				for (int i = 0; i < X1s.size(); i++) {
					double[] UandV = UsAndVs(weightList.get(weightLoc), X1s.get(i), X2s.get(i));
					double[] partialD = partialDerivatives(UandV, ys.get(i), weightList.get(weightLoc));
					double[] partialDW = partialDerivativesWeights(UandV, partialD, X1s.get(i), X2s.get(i));
					double[] newWeights = updateWeights(partialDW, weightList.get(weightLoc), N);
					weights = newWeights;
					weightList.add(newWeights);
					weightLoc += 1;
				}
				// Print out Values
				for (double weight : weightList.get(weightLoc)) {
					System.out.print(String.format("%.5f", weight) + " ");
				}
				System.out.println();
				double evalSetError = getError(EvalX1s, EvalX2s, Evalys, weightList.get(weightLoc));
				System.out.println(String.format("%.5f", evalSetError));
			}
		}
		if (flag == 800) {
			double N = Double.valueOf(args[10]);
			double T = Double.valueOf(args[11]);
			// Gets List of X1s X2s and Y values
			ArrayList<ArrayList<Double>> EvalList = readFile("./hw2_midterm_A_eval.txt");
			ArrayList<ArrayList<Double>> TestList = readFile("./hw2_midterm_A_test.txt");
			ArrayList<ArrayList<Double>> TrainList = readFile("./hw2_midterm_A_train.txt");
			ArrayList<Double> X1s = TrainList.get(0);
			ArrayList<Double> X2s = TrainList.get(1);
			ArrayList<Double> ys = TrainList.get(2);
			ArrayList<Double> EvalX1s = EvalList.get(0);
			ArrayList<Double> EvalX2s = EvalList.get(1);
			ArrayList<Double> Evalys = EvalList.get(2);
			ArrayList<Double> testX1s = TestList.get(0);
			ArrayList<Double> testX2s = TestList.get(1);
			ArrayList<Double> testYs = TestList.get(2);
			// Initializes other Variables
			ArrayList<Double> setError = new ArrayList<>();
			ArrayList<double[]> weightList = new ArrayList<double[]>();
			weightList.add(weights);
			int weightLoc = 0;
			boolean done = false;
			for (int t = 0; t < T && !done; t++) {
				// Stochastic Gradient Descent
				for (int i = 0; i < X1s.size() && !done; i++) {
					double[] UandV = UsAndVs(weightList.get(weightLoc), X1s.get(i), X2s.get(i));
					double[] partialD = partialDerivatives(UandV, ys.get(i), weightList.get(weightLoc));
					double[] partialDW = partialDerivativesWeights(UandV, partialD, X1s.get(i), X2s.get(i));
					double[] newWeights = updateWeights(partialDW, weightList.get(weightLoc), N);
					weights = newWeights;
					weightList.add(newWeights);
					weightLoc += 1;
				}
				double evalSetError = getError(EvalX1s, EvalX2s, Evalys, weightList.get(weightLoc));
				setError.add(evalSetError);
				if (t != 0) {
					if (setError.get(t) > setError.get(t - 1)) {
						done = true;
					}
				}
				// Print out Values
				if (done) {
					System.out.println(t + 1);
					for (double weight : weightList.get(weightLoc)) {
						System.out.print(String.format("%.5f", weight) + " ");
					}
					System.out.println();
					System.out.println(String.format("%.5f", setError.get(t)));
					double correctPredictions = makePredictions(testX1s, testX2s, testYs, weightList.get(weightLoc));
					double classAccuracy = (correctPredictions / testX1s.size());
					System.out.println(String.format("%.5f",classAccuracy));
					break;
				}
			}
		}
	}

	private static double makePredictions(ArrayList<Double> testX1s, ArrayList<Double> testX2s,ArrayList<Double> testYs, double[] weights) {
		double correctPredictions = 0;
		double prediction = 0;
		for (int i = 0; i < testX1s.size(); i++) {
			double[] UandV = UsAndVs(weights, testX1s.get(i), testX2s.get(i));
			prediction = UandV[5];
			if ((prediction < .5 && testYs.get(i) == 0) || (prediction >= .5 && testYs.get(i) == 1)) {
				correctPredictions +=1;
			}
		}
		return correctPredictions;
	}

	private static double getError(ArrayList<Double> evalX1s, ArrayList<Double> evalX2s, ArrayList<Double> evalys,
			double[] weights) {
		double evalError = 0;
		for (int i = 0; i < evalys.size(); i++) {
			double[] UandV = UsAndVs(weights, evalX1s.get(i), evalX2s.get(i));
			evalError += (.5) * (Math.pow((UandV[5] - evalys.get(i)), 2));
		}
		return evalError;
	}

	private static double[] updateWeights(double[] partialDW, double[] weights, double N) {
		double[] newWeight = new double[9];
		for (int i = 0; i < weights.length; i++) {
			newWeight[i] = weights[i] - N * (partialDW[i]);
		}
		return newWeight;
	}

	private static double[] partialDerivativesWeights(double[] UandV, double[] partialD, double x1, double x2) {
		double[] partialDW = new double[9];
		partialDW[0] = (1 * partialD[3]);
		partialDW[1] = (x1 * partialD[3]);
		partialDW[2] = (x2 * partialD[3]);
		partialDW[3] = (1 * partialD[5]);
		partialDW[4] = (x1 * partialD[5]);
		partialDW[5] = (x2 * partialD[5]);
		partialDW[6] = (1 * partialD[1]);
		partialDW[7] = (UandV[1] * partialD[1]);
		partialDW[8] = (UandV[3] * partialD[1]);
		return partialDW;
	}

	private static double[] UsAndVs(double[] w, double x1, double x2) {
		double[] UandV = new double[6];
		UandV[0] = (w[0] * 1 + w[1] * x1 + w[2] * x2);
		UandV[1] = ReLU(UandV[0]);
		UandV[2] = (w[3] * 1 + w[4] * x1 + w[5] * x2);
		UandV[3] = ReLU(UandV[2]);
		UandV[4] = (w[6] * 1 + w[7] * UandV[1] + w[8] * UandV[3]);
		UandV[5] = sigmoid(UandV[4]);
		return UandV;
	}

	private static double[] partialDerivatives(double[] UandV, double y, double[] w) {
		double[] partialD = new double[6];
		double error = (.5) * (Math.pow((UandV[5] - y), 2));
		// Partial output layer Vc
		partialD[0] = UandV[5] - y;
		// partial intermediate layer Uc
		partialD[1] = partialD[0] * (UandV[5] * (1 - UandV[5]));
		// partial hidden layer Va
		partialD[2] = (w[7] * partialD[1]);
		// partial hidden layer Ua
		if (UandV[0] < 0) {
			partialD[3] = 0;
		} else {
			partialD[3] = partialD[2];
		}
		// partial hidden layer Vb
		partialD[4] = (w[8] * partialD[1]);
		// parital hidden layer Ub
		if (UandV[2] < 0) {
			partialD[5] = 0;
		} else {
			partialD[5] = partialD[4];
		}
		return partialD;
	}

	private static double sigmoid(double Uc) {
		double Vc = 1 / (1 + Math.pow(Math.E, -Uc));
		return Vc;
	}

	private static double ReLU(double d) {
		double V = Math.max(d, 0);
		return V;
	}

	private static ArrayList<ArrayList<Double>> readFile(String file) {
		ArrayList<Double> X1 = new ArrayList<Double>();
		ArrayList<Double> X2 = new ArrayList<Double>();
		ArrayList<Double> Y = new ArrayList<Double>();
		ArrayList<ArrayList<Double>> XsAndYs = new ArrayList<ArrayList<Double>>();
		boolean x1 = true;
		boolean x2 = false;
		boolean y = false;
		try {
			File f = new File(file);
			Scanner sc = new Scanner(f);
			while (sc.hasNext()) {
				if (sc.hasNextDouble() && x1) {
					double i = sc.nextDouble();
					X1.add(i);
					x1 = false;
					x2 = true;
				} else if (sc.hasNextDouble() && x2) {
					double i = sc.nextDouble();
					X2.add(i);
					x2 = false;
					y = true;
				} else if (sc.hasNextDouble() && y) {
					double i = sc.nextDouble();
					Y.add(i);
					y = false;
					x1 = true;
				} else {
					sc.next();
				}
			}
		} catch (FileNotFoundException ex) {
			System.out.println("File Not Found.");
		}
		XsAndYs.add(X1);
		XsAndYs.add(X2);
		XsAndYs.add(Y);
		return XsAndYs;
	}
}
