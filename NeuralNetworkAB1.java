/*
 * Author: Robinson Xiang
 * Date Created: January 30th, 2026
 *
 * This Java class implements a A-B-1 neural network. The program supports two independent runtime modes:
 *   - Training mode, which trains the network using gradient descent on single-output error
 *   - Running mode, which runs the network using fixed or randomly generated weights with forward propagation
 */
public class NeuralNetworkAB1
{
   int A;                           // number of input layer nodes
   int B;                           // number of hidden layer nodes
   boolean doTrain, doRun;          // doTrain is true when training; doRun is true when running

   int weightsMode;                 // 0 -> manual weights, 1 -> random weights

   boolean printInputs, printTruth; // printInputs prints inputTable; printTruth prints truthTable

   double randLow, randHigh;        // minimum and maximum random weight values
   double lambda;                   // learning rate in gradient descent
   int maxIterations;               // max number of passes during training
   double errorThreshold;           // threshold for average error to stop training

   double[] a;                      // input values
   double[] h;                      // hidden activations
   double F0;                       // output value

   double[] hiddenSums;             // Θj, the hidden pre-activation sums
   double outputSum;                // Θ0, the output pre-activation sum

   double[][] inputWeights;         // wkj, input to hidden weights
   double[] outputWeights;          // wj0, hidden to output weights

   int numCases;                    // number of test cases
   double[][] inputTable;           // input value for case c at input index k
   double[] truthTable;             // stores the correct outputs for inputTable

   double[][] inputDeltaWeights;    // Δwkj = −λ (∂E/∂wkj)
   double[] outputDeltaWeights;     // Δwj0 = −λ (∂E/∂wj0)
   double[] hiddenErrorSignal;      // Ψj = Ωj f'(Θj)

   int iterations;                  // number of passes completed during training
   double avgError;                 // average error over all cases for the most recent pass
   boolean stopByError;             // true if avgError <= errorThreshold; false otherwise
   boolean stopByMax;               // true if iterations >= maxIterations; false otherwise

/**
 * Sets and initalizes configuration parameters with values.
 */
   public void setConfigurationParameters()
   {
      doTrain = true;
      doRun = true;

      A = 3;
      B = 3;
      numCases = 4;                 
      weightsMode = 1;              // 0 -> manual, 1 -> random

      printInputs = true;
      printTruth = false;

      randLow = -1.5;
      randHigh = 1.5;

      lambda = 0.3;
      maxIterations = 100000;
      errorThreshold = 2e-4;

      return;
   } // public void setConfigurationParameters()

/**
 * Prints configuration parameters.
 */
   public void echoConfigurationParameters()
   {
      System.out.println("Network Configuration: " + A + "-" + B + "-1");
      System.out.println("Training = " + doTrain + ", Running = " + doRun);
      System.out.println("Weights = " + (weightsMode == 0 ? "manual" : "random"));

      if (doTrain)
      {
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
      a = new double[A];                       // a[k]
      h = new double[B];                       // h[j]
      hiddenSums = new double[B];              // Θj

      inputWeights = new double[A][B];         // wkj
      outputWeights = new double[B];           // wj0

      if (doTrain || doRun || printInputs || printTruth)
      {
         inputTable = new double[numCases][A];
      } 

      if (doTrain || printTruth)
      {
         truthTable = new double[numCases];
      } 

      if (doTrain)
      {
         inputDeltaWeights = new double[A][B]; // Δwkj
         outputDeltaWeights = new double[B];   // Δwj0
         hiddenErrorSignal = new double[B];    // Ψj
      } 

      return;
   } // public void allocateArrays()

/**
 * Populates tables and initializes weights either manually or randomly.
 */
   public void populateArrays()
   {
      if (inputTable != null)
      {
         populateInputTable();
      }

      if (truthTable != null)
      {
         populateTruthTable();
      }

      if (weightsMode == 0) setManualWeights();
      else randomizeWeights(randLow, randHigh);

      return;
   } // public void populateArrays()
   

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
   } 

/**
 * Populates truthTable. Manually configurable.
 */
   private void populateTruthTable()
   {
      truthTable[0] = 0.0;
      truthTable[1] = 1.0;
      truthTable[2] = 1.0;
      truthTable[3] = 0.0;
      return;
   } 

/**
 * Sets the weights manually for a known configuration.
 */
   public void setManualWeights()
   {
      for (int k = 0; k < A; k++)
      {
         for (int j = 0; j < B; j++)
         {
            inputWeights[k][j] = 0.0;
         } // for (int j = 0; j < B; j++)
      } // for (int k = 0; k < A; k++)

      if (A == 2 && B == 2)               // Manual weights for 2-2-1
      {
         inputWeights[0][0] = 0.324;      // w00
         inputWeights[0][1] = 0.341;      // w01
         inputWeights[1][0] = 0.0;        // w10
         inputWeights[1][1] = 0.0;        // w11
         outputWeights[0] = 1.0;          // w00
         outputWeights[1] = 0.0;          // w10
      }

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
      for (int j = 0; j < B; j++)
      {
         outputWeights[j] = randomNumber(low, high);
      } 

      for (int k = 0; k < A; k++)
      {
         for (int j = 0; j < B; j++)
         {
            inputWeights[k][j] = randomNumber(low, high);
         } // for (int j = 0; j < B; j++)
      } // for (int k = 0; k < A; k++)

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
 * Runs the neural network once for running.
 *
 * @return output activation F0.
 */
   public double run()
   {
      for (int j = 0; j < B; j++)
      {
         double sum = 0.0;                    // Θj

         for (int k = 0; k < A; k++)
         {
            sum += a[k] * inputWeights[k][j]; // Θj += ak * wkj
         } // for (int k = 0; k < A; k++)

         hiddenSums[j] = sum;
         h[j] = sigmoid(sum);                 // f(Θj)
      } // for (int j = 0; j < B; j++)

      double sum0 = 0.0; // Θ0
      for (int j = 0; j < B; j++)
      {
         sum0 += h[j] * outputWeights[j];     // Θ0 += hj * wj0
      }

      outputSum = sum0;
      F0 = sigmoid(sum0);                     // f(Θ0)

      return F0;
   } // public double run()

/**
 * Runs the neural network once, designed for training only. Identical to run() above for now.
 *
 * @return output activation F0.
 */
   public double runForTraining()
   {
      for (int j = 0; j < B; j++)
      {
         double sum = 0.0;                    // Θj

         for (int k = 0; k < A; k++)
         {
            sum += a[k] * inputWeights[k][j]; // Θj += ak * wkj
         } // for (int k = 0; k < A; k++)

         hiddenSums[j] = sum;
         h[j] = sigmoid(sum);                 // f(Θj)
      } // for (int j = 0; j < B; j++)

      double sum0 = 0.0; // Θ0
      for (int j = 0; j < B; j++)
      {
         sum0 += h[j] * outputWeights[j];     // Θ0 += hj * wj0
      }

      outputSum = sum0;
      F0 = sigmoid(sum0);                     // f(Θ0)

      return F0;
   } // public double runForTraining()


/**
 * Sigmoid activation function f(x) = 1 / (1 + e^(-x)).
 *
 * @param x input value
 * @return sigmoid output f(x).
 */
   private double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // private double sigmoid(double x)

/**
 * Computes the derivative of the sigmoid function evaluated at theta.
 *
 * @param theta pre-activation sum theta
 * @return sigmoid derivative f'(theta).
 */
   private double derivativeSigmoid(double theta)
   {
      double e = Math.exp(-theta);
      double denom = 1.0 + e;
      return e / (denom * denom);
   } // private double derivativeSigmoidTheta(double theta)

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
         } // while (!stopByError && !stopByMax)
      } // if (doTrain)

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

         double output = run();
         double target = truthTable[c];

         double outputError = (target - output);         // ω0 = (T0 − F0)
         sumErrors += 0.5 * outputError * outputError;   // E = 1/2 * ω0^2

         computeWeights(target, output);                 // Compute Δw arrays
         applyWeights();                                 // Apply Δw arrays to weights
      } // for (int c = 0; c < numCases; c++)

      return sumErrors / (double)numCases;
   } // private double trainOnce()

/**
 * Loads one input case row c from inputTable into the input array a[k].
 *
 * @param c case index
 */
   private void loadCaseInputs(int c)
   {
      for (int k = 0; k < A; k++)
      {
         a[k] = inputTable[c][k];
      } // for (int k = 0; k < A; k++)

      return;
   } // private void loadCaseInputs(int c)

/**
 * Computes delta weights (Δw) for one training case using gradient descent.
 *
 * @param target target output
 * @param out    output value
 */
   private void computeWeights(double target, double out)
   {
      double outputError = (target - out);                                          // ω0 = (T0 − F0)
      double outputDelta = outputError * derivativeSigmoid(outputSum);              // ψ0 = ω0 f'(Θ0)

      for (int j = 0; j < B; j++)
      {
         double gradOutputWeight = -h[j] * outputDelta;                       // ∂E/∂wj0 = −hj ψ0
         outputDeltaWeights[j] = -lambda * gradOutputWeight;                  // Δwj0 = −λ (∂E/∂wj0)
      } 

      for (int j = 0; j < B; j++)
      {
         double hiddenOutputWeight = outputDelta * outputWeights[j];          // Ωj = ψ0 wj0
         hiddenErrorSignal[j] = hiddenOutputWeight * derivativeSigmoid(hiddenSums[j]); // Ψj = Ωj f'(Θj)

         for (int k = 0; k < A; k++)
         {
            double gradInputWeight = -a[k] * hiddenErrorSignal[j];            // ∂E/∂wkj = −ak Ψj
            inputDeltaWeights[k][j] = -lambda * gradInputWeight;              // Δwkj = −λ (∂E/∂wkj)
         } // for (int k = 0; k < A; k++)
      } // for (int j = 0; j < B; j++)

      return;
   } // private void computeWeights(double target, double out)

/**
 * Applies the delta weights computed by computeWeights() to the weight arrays.
 */
   private void applyWeights()
   {
      for (int j = 0; j < B; j++)
      {
         outputWeights[j] += outputDeltaWeights[j];
      } 

      for (int k = 0; k < A; k++)
      {
         for (int j = 0; j < B; j++)
         {
            inputWeights[k][j] += inputDeltaWeights[k][j];
         } // for (int j = 0; j < B; j++)
      } // for (int k = 0; k < A; k++)

      return;
   } // private void applyWeights()

/**
 * Prints training exit information.
 */
   public void reportTrainingResults()
   {
      if (doTrain)
      {
         System.out.println("Training Exit");
         System.out.println("Passed Error Threshold = " + stopByError + ", Reached Max Iterations = " + stopByMax);
         System.out.println("Average Error = " + String.format("%.6f", avgError) + ", Iterations = " + iterations);
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
   public double[] runAllCases()
   {
      double[] outputs = null;

      if (doRun)
      {
         outputs = new double[numCases];

         for (int c = 0; c < numCases; c++)
         {
            loadCaseInputs(c);
            outputs[c] = run();
         } // for (int c = 0; c < numCases; c++)
      } // if (doRun)

      return outputs;
   } // public double[] runAllCases()

/**
 * Reports run results.
 *
 * @param outputs the output array returned by runAllCases().
 */
   public void reportRunResults(double[] outputs)
   {
      if (doRun && outputs != null)
      {
         System.out.println("Run Results");

         if (printInputs)
         {
            System.out.println("Inputs:");
            for (int c = 0; c < numCases; c++)
            {
               for (int k = 0; k < A; k++)
               {
                  System.out.print((int)inputTable[c][k]);
                  if (k < A - 1) System.out.print(" ");
               } // for (int k = 0; k < A; k++)
               System.out.println();
            } // for (int c = 0; c < numCases; c++)
         } // if (printInputs)

         if (printTruth && truthTable != null)
         {
            System.out.println("Truth Table:");
            for (int c = 0; c < numCases; c++)
            {
               System.out.println((int)truthTable[c]);
            } // for (int c = 0; c < numCases; c++)
         } // if (printTruth && truthTable != null)

         System.out.println("Outputs:");
         for (int c = 0; c < numCases; c++)
         {
            System.out.println(String.format("%.6f", outputs[c]));
         } 

         System.out.println();
      } // if (doRun && outputs != null)

      return;
   } // public void reportRunResults(double[] outputs)

/**
 * Runs the program. Initializes and prints configuration,
 * populates arrays and weights, and runs or trains (or both) depending on user configuration.
 */
   public static void main(String[] args)
   {
      NeuralNetworkAB1 nn = new NeuralNetworkAB1();

      nn.setConfigurationParameters();
      nn.echoConfigurationParameters();

      nn.allocateArrays();
      nn.populateArrays();

      nn.trainAllCases();
      nn.reportTrainingResults();

      double[] outputs = nn.runAllCases();
      nn.reportRunResults(outputs);

      return;
   } // public static void main(String[] args)
} // public class NeuralNetworkAB1