import java.io.*;

/*
 * Author: Robinson Xiang
 * Date Created: February 23rd, 2026
 *
 * This Java class implements a A-B-C neural network. The program supports two independent runtime modes:
 *   - Training mode, which trains the network using gradient descent
 *   - Running mode, which runs the network using fixed, loaded, or randomly generated weights
 */
public class NeuralNetworkABC
{
   int numInputActivations;            // number of input layer activations
   int numHiddenActivations;           // number of hidden layer activations
   int numOutputActivations;           // number of output layer activations

   boolean doTrain, doRun;             // doTrain is true when training; doRun is true when running

   int weightsMode;                    // 0 -> manual weights, 1 -> random weights
   boolean loadWeights;                // true -> load weights from binary file
   boolean saveWeights;                // true -> save weights to binary file

   String inputWeights;                // binary file to load weights from
   String outputWeights;               // binary file to save weights to

   boolean printInputs, printTruth;    // printInputs prints inputTable; printTruth prints truthTable
   boolean printOutputs;               // prints actual network outputs

   double randLow, randHigh;           // minimum and maximum random weight values
   double lambda;                      // learning rate in gradient descent
   int maxIterations;                  // max number of passes during training
   double errorThreshold;              // threshold for average error to stop training

   double[] a;                         // input values a[k]
   double[] h;                         // hidden activations h[j]
   double[] F;                         // output activations F[i]

   double[] thetaJ;                   // Theta_j[j], hidden pre-activation sums
   double[] thetaI;                   // Theta_i[i], output pre-activation sums

   double[][] wkj;                     // wkj, input to hidden weights
   double[][] wji;                     // wji, hidden to output weights

   int numCases;                       // number of test cases
   double[][] inputTable;              // stores input values for each case
   double[][] truthTable;              // stores correct outputs for each case

   double[][] deltaWkj;                // Delta wkj
   double[][] deltaWji;                // Delta wji
   double[] omega;                     // omega[i]
   double[] psi;                       // psi[i]
   double[] Omega;                     // Omega[j]
   double[] Psi;                       // Psi[j]

   int iterations;                     // number of passes completed during training
   double avgError;                    // average error over all cases for the most recent pass
   boolean stopByError;                // true if avgError <= errorThreshold
   boolean stopByMax;                  // true if iterations >= maxIterations

   long trainingTime;                // time to train
   long runningTime;                 // time to run

/**
 * Sets and initializes configuration parameters with values.
 */
   public void setConfigurationParameters()
   {
      doTrain = false;
      doRun = false;

      numInputActivations = 2;
      numHiddenActivations = 2;
      numOutputActivations = 3;
      numCases = 4;

      weightsMode = 1;                 // 0 -> manual, 1 -> random, 2 -> load
      loadWeights = true;
      saveWeights = true;

      inputWeights = "weightsInput.bin";
      outputWeights = "weightsOutput.bin";

      printInputs = true;
      printTruth = true;
      printOutputs = true;

      randLow = 0.1;
      randHigh = 1.5;

      lambda = 0.3;
      maxIterations = 40000;
      errorThreshold = 2e-4;

      trainingTime = 0;
      runningTime = 0;

      return;
   } // public void setConfigurationParameters()

/**
 * Prints configuration parameters.
 */
   public void echoConfigurationParameters()
   {
      System.out.println("Network Configuration: "
         + numInputActivations + "-" + numHiddenActivations + "-" + numOutputActivations);
      System.out.println("Training = " + doTrain + ", Running = " + doRun);
      System.out.println("Weights = " + (loadWeights ? "binary file" : (weightsMode == 0 ? "manual" : "random")));
      System.out.println("Input Weights File = " + inputWeights);
      System.out.println("Output Weights File = " + outputWeights);

      if (doTrain)
      {
         System.out.println("\nRuntime Training Parameters");
         System.out.println("Random[" + randLow + ", " + randHigh + "]");
         System.out.println("Lambda = " + lambda);
         System.out.println("Max Iterations = " + maxIterations);
         System.out.println("Error Threshold = " + errorThreshold);
      }

      System.out.println();
      return;
   } // public void echoConfigurationParameters()

/**
 * Initializes the arrays for the network, cases, and training.
 */
   public void allocateArrays()
   {
      a = new double[numInputActivations];                           // a[k]
      h = new double[numHiddenActivations];                          // h[j]
      F = new double[numOutputActivations];                          // F[i]

      wkj = new double[numInputActivations][numHiddenActivations];   // wkj
      wji = new double[numHiddenActivations][numOutputActivations];  // wji

      inputTable = new double[numCases][numInputActivations];

      if (doTrain || printTruth)
      {
         truthTable = new double[numCases][numOutputActivations];
      }

      if (doTrain)
      {
         thetaJ = new double[numHiddenActivations];
         thetaI = new double[numOutputActivations];

         deltaWkj = new double[numInputActivations][numHiddenActivations];
         deltaWji = new double[numHiddenActivations][numOutputActivations];

         omega = new double[numOutputActivations];
         psi = new double[numOutputActivations];
         Omega = new double[numHiddenActivations];
         Psi = new double[numHiddenActivations];
      }
      return;
   } // public void allocateArrays()

/**
 * Populates tables and initializes weights either manually, randomly, or from file.
 */
   public void populateArrays() throws IOException
   {
      if (inputTable != null)
      {
         populateInputTable();
      }

      if (truthTable != null)
      {
         populateTruthTable();
      }

      initializeWeights();
      return;
   } // public void populateArrays() throws IOException

/**
 * Initializes weights.
 */
   private void initializeWeights() throws IOException
   {
      if (loadWeights)
      {
         loadWeights(inputWeights);
         return;
      }

      if (weightsMode == 0)
      {
         setManualWeights();
         return;
      }

      randomizeWeights(randLow, randHigh);
      return;
   } // private void initializeWeights() throws IOException

/**
 * Populates inputTable.
 */
   private void populateInputTable()
   {
      inputTable[0][0] = 0.0; inputTable[0][1] = 0.0;
      inputTable[1][0] = 0.0; inputTable[1][1] = 1.0;
      inputTable[2][0] = 1.0; inputTable[2][1] = 0.0;
      inputTable[3][0] = 1.0; inputTable[3][1] = 1.0;

      return;
   } // private void populateInputTable()

/**
 * Populates truthTable. Manually configurable but preconfigured with outputs AND, OR, XOR.
 */
   private void populateTruthTable()
   {
      truthTable[0][0] = 0.0; truthTable[0][1] = 0.0; truthTable[0][2] = 0.0;
      truthTable[1][0] = 0.0; truthTable[1][1] = 1.0; truthTable[1][2] = 1.0;
      truthTable[2][0] = 0.0; truthTable[2][1] = 1.0; truthTable[2][2] = 1.0;
      truthTable[3][0] = 1.0; truthTable[3][1] = 1.0; truthTable[3][2] = 0.0;

      return;
   } // private void populateTruthTable()

/**
 * Sets the weights manually for a known configuration.
 */
   public void setManualWeights()
   {
      wkj[0][0] = 0.324;          // w00
      wkj[0][1] = 0.341;          // w01
      wkj[1][0] = 0.23;           // w10
      wkj[1][1] = 0.15;           // w11

      wji[0][0] = 0.76;           // w00
      wji[0][1] = 0.23;           // w10
      wji[1][0] = 0.76;           // w00
      wji[1][1] = 0.98;           // w10
      return;
   } // public void setManualWeights()

/**
 * Randomizes all weights from low to high.
 *
 * @param low  the minimum random weight value.
 * @param high the maximum random weight value.
 */
   public void randomizeWeights(double low, double high)
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            wkj[k][j] = randomNumber(low, high);
         }
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            wji[j][i] = randomNumber(low, high);
         }
      }
      return;
   } // public void randomizeWeights(double low, double high)

/**
 * Generates a random number in the range from low to high.
 *
 * @param low  the minimum random weight value.
 * @param high the maximum random weight value.
 * @return r random value between low and high.
 */
   private double randomNumber(double low, double high)
   {
      double r01 = Math.random();
      double r = low + r01 * (high - low);

      return r;
   } // private double randomNumber(double low, double high)

/**
 * Saves weights in binary.
 *
 * @param fileName binary output file name
 * @throws IOException if write fails
 */
   public void saveWeights(String fileName) throws IOException
   {
      DataOutputStream out = new DataOutputStream(
      new BufferedOutputStream(new FileOutputStream(fileName)));

      out.writeInt(numInputActivations);
      out.writeInt(numHiddenActivations);
      out.writeInt(numOutputActivations);

      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            out.writeDouble(wkj[k][j]);
         }
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            out.writeDouble(wji[j][i]);
         }
      }

      out.close();
      return;
   } // public void saveWeights(String fileName) throws IOException

/**
 * Loads weights in binary.
 *
 * @param fileName binary input file name
 * @throws IOException if read fails 
 */
   public void loadWeights(String fileName) throws IOException
   {
      DataInputStream in = new DataInputStream(
         new BufferedInputStream(new FileInputStream(fileName)));

      int fileInputs = in.readInt();
      int fileHidden = in.readInt();
      int fileOutputs = in.readInt();

      if (fileInputs != numInputActivations
         || fileHidden != numHiddenActivations
         || fileOutputs != numOutputActivations)
      {
         in.close();
         throw new IOException("Weight file configuration does not match current network.");
      }

      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            wkj[k][j] = in.readDouble();
         }
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            wji[j][i] = in.readDouble();
         }
      }

      in.close();
      return;
   } // public void loadWeights(String fileName) throws IOException

/**
 * Generic activation function.
 *
 * @param theta pre-activation sum
 * @return activation value
 */
   private double activationFunction(double theta)
   {
      return sigmoid(theta);
   } // private double activationFunction(double theta)

/**
 * Generic derivative function.
 *
 * @param theta pre-activation sum
 * @return derivative value
 */
   private double derivativeActivationFunction(double theta)
   {
      return derivativeSigmoid(theta);
   } // private double derivativeActivationFunction(double theta)

/**
 * Calculates sigmoid activation function 1 / (1 + e^(-x)) with input value x.
 *
 * @param x input value
 * @return output of sigmoid
 */
   private double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // private double sigmoid(double x)

/**
 * Computes the derivative of the sigmoid function.
 *
 * @param theta pre-activation sum theta
 * @return derivative of sigmoid.
 */
   private double derivativeSigmoid(double theta)
   {
      double fTheta = sigmoid(theta);
      return sigmoid(theta) * (1.0 - fTheta);
   } // private double derivativeSigmoid(double theta)

/**
 * Runs the neural network once for running only.
 */
   public void run()
   {
      for (int j = 0; j < numHiddenActivations; j++)
      {
         double sum = 0.0;

         for (int k = 0; k < numInputActivations; k++)
         {
            sum += a[k] * wkj[k][j];
         }

         h[j] = activationFunction(sum);
      }

      for (int i = 0; i < numOutputActivations; i++)
      {
         double sum = 0.0;

         for (int j = 0; j < numHiddenActivations; j++)
         {
            sum += h[j] * wji[j][i];
         }

         F[i] = activationFunction(sum);
      }
      return;
   } // public void run()

/**
 * Runs the neural network once for training only.
 */
   public void runForTraining()
   {
      for (int j = 0; j < numHiddenActivations; j++)
      {
         thetaJ[j] = 0.0;

         for (int k = 0; k < numInputActivations; k++)
         {
            thetaJ[j] += a[k] * wkj[k][j];
         }

         h[j] = activationFunction(thetaJ[j]);
      }

      for (int i = 0; i < numOutputActivations; i++)
      {
         thetaI[i] = 0.0;

         for (int j = 0; j < numHiddenActivations; j++)
         {
            thetaI[i] += h[j] * wji[j][i];
         }

         F[i] = activationFunction(thetaI[i]);
      }
      return;
   } // public void runForTraining()

/**
 * Trains using all training cases until a stopping condition is met.
 * Stopping conditions are not mutually exclusive.
 */
   public void trainAllCases()
   {
      if (doTrain)
      {
         iterations = 0;
         avgError = 0.0;
         stopByError = false;
         stopByMax = false;

         while (!stopByError && !stopByMax)
         {
            iterations++;
            avgError = trainOnce();

            if (avgError <= errorThreshold) stopByError = true;
            if (iterations >= maxIterations) stopByMax = true;
         }
      }
      return;
   } // public void trainAllCases()

/**
 * Performs one full pass over all training cases.
 *
 * @return average error across all cases for this pass.
 */
   private double trainOnce()
   {
      double sumErrors = 0.0;

      for (int c = 0; c < numCases; c++)
      {
         loadCaseInputs(c);
         runForTraining();
         sumErrors += computeWeights(c);
         applyWeights();
      }
      return sumErrors / (double)numCases;
   } // private double trainOnce()

/**
 * Loads one input case row c from inputTable into the input array a[k].
 *
 * @param c case index
 */
   private void loadCaseInputs(int c)
   {
      for (int k = 0; k < numInputActivations; k++)
      {
         a[k] = inputTable[c][k];
      }
      return;
   } // private void loadCaseInputs(int c)

/**
 * Computes delta weights (Δw) for one training case using gradient descent.
 *
 * @param caseIndex index of the current training case
 * @return error for this case
 */
   private double computeWeights(int caseIndex)
   {
      double caseError = 0.0;

      for (int i = 0; i < numOutputActivations; i++)
      {
         omega[i] = truthTable[caseIndex][i] - F[i];
         caseError += 0.5 * omega[i] * omega[i];
         psi[i] = omega[i] * derivativeActivationFunction(thetaI[i]);
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         Omega[j] = 0.0;

         for (int i = 0; i < numOutputActivations; i++)
         {
            Omega[j] += psi[i] * wji[j][i];
         }

         Psi[j] = Omega[j] * derivativeActivationFunction(thetaJ[j]);
      }

      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            deltaWji[j][i] = lambda * h[j] * psi[i];
         }
      }

      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            deltaWkj[k][j] = lambda * a[k] * Psi[j];
         }
      }
      return caseError;
   } // private double computeWeights(int caseIndex)

/**
 * Applies the delta weights computed by computeWeights() to the weight arrays.
 */
   private void applyWeights()
   {
      for (int j = 0; j < numHiddenActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            wji[j][i] += deltaWji[j][i];
         }
      }

      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenActivations; j++)
         {
            wkj[k][j] += deltaWkj[k][j];
         }
      }
      return;
   } // private void applyWeights()

/**
 * Prints training exit information.
 */
   public void reportTrainingResults()
   {
      if (doTrain)
      {
         System.out.println("Training Exit Information");
         System.out.println("Exit Reason = "
            + (stopByError ? "Error Threshold Reached" : "Maximum Iterations Reached"));
         System.out.println("Iterations = " + iterations);
         System.out.println("Error = " + avgError);
         System.out.println("Time to Train = " + trainingTime + " ms");
         System.out.println();
      }
      return;
   } // public void reportTrainingResults()

/**
 * Runs all cases by loading input rows into a[k] then calling run().
 * Does not require truthTable.
 *
 * @return array of outputs, null if doRun is false.
 */
   public double[][] runAllCases()
   {
      double[][] outputs = null;

      if (doRun)
      {
         outputs = new double[numCases][numOutputActivations];

         for (int c = 0; c < numCases; c++)
         {
            loadCaseInputs(c);
            run();

            for (int i = 0; i < numOutputActivations; i++)
            {
               outputs[c][i] = F[i];
            }
         }
      }
      return outputs;
   } // public double[][] runAllCases()

/**
 * Reports run results.
 *
 * @param outputs the output array returned by runAllCases().
 */
   public void reportRunResults(double[][] outputs)
   {
      if (doRun && outputs != null)
      {
         System.out.println("Running Exit Information");
         System.out.println("Time to Run = " + runningTime + " ms");
         System.out.println();

         System.out.println("Input" + (printTruth ? "    Truth Table" : "") + "    Output");
         for (int c = 0; c < numCases; c++)
         {
            if (printInputs)
            {
               for (int k = 0; k < numInputActivations; k++)
               {
                  System.out.print((int)inputTable[c][k]);
                  if (k < numInputActivations - 1) System.out.print(" ");
               }
               System.out.print("      ");
            }

            if (printTruth && truthTable != null)
            {
               for (int i = 0; i < numOutputActivations; i++)
               {
                  System.out.print((int)truthTable[c][i]);
                  if (i < numOutputActivations - 1) System.out.print(" ");
               }
               System.out.print("           ");
            }

            if (printOutputs)
            {
               for (int i = 0; i < numOutputActivations; i++)
               {
                  System.out.print(String.format("%.6f", outputs[c][i]));
                  if (i < numOutputActivations - 1) System.out.print(" ");
               }
            }

            System.out.println();
         }

         System.out.println();
      }

      return;
   } // public void reportRunResults(double[][] outputs)

/**
 * Runs the program. Initializes and prints configuration,
 * populates arrays and weights, and runs or trains (or both) depending on user configuration.
 */
   public static void main(String[] args) throws IOException
   {
      NeuralNetworkABC nn = new NeuralNetworkABC();

      nn.setConfigurationParameters();
      nn.echoConfigurationParameters();

      nn.allocateArrays();
      nn.populateArrays();

      if (nn.doTrain)
      {
         long startTrain = System.currentTimeMillis();
         nn.trainAllCases();
         long endTrain = System.currentTimeMillis();
         nn.trainingTime = endTrain - startTrain;

         nn.reportTrainingResults();

         if (nn.saveWeights)
         {
            nn.saveWeights(nn.outputWeights);
         }
      }

      double[][] outputs = null;

      if (nn.doRun)
      {
         long startRun = System.currentTimeMillis();
         outputs = nn.runAllCases();
         long endRun = System.currentTimeMillis();
         nn.runningTime = endRun - startRun;

         nn.reportRunResults(outputs);
      }

      return;
   } // public static void main(String[] args)
} // public class NeuralNetworkABC