using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetwork
{

    #region Transfer Functions and derivatives

    public enum TransferFunction
    {
        None,
        Sigmoid,
        Linear,
        Gaussian,
        RationalSigmoid
    }

    public static class TransferFunctions
    {
        public static double Evaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid(input);
                case TransferFunction.Linear:
                    return linear(input);
                case TransferFunction.Gaussian:
                    return gaussian(input);
                case TransferFunction.RationalSigmoid:
                    return rationalSigmoid(input);
                case TransferFunction.None:
                default:
                    return 0.0;
            }
        }
        public static double EvaluateDerivative(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoidDerivative(input);
                case TransferFunction.Linear:
                    return linearDerivative(input);
                case TransferFunction.Gaussian:
                    return gaussianDerivative(input);
                case TransferFunction.RationalSigmoid:
                    return rationalSigmoidDerivative(input);
                case TransferFunction.None:
                default:
                    return 0.0;
            }
        }

        /* Transfer Functions */
        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double sigmoidDerivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        private static double linear(double x)
        {
            return x;
        }

        private static double linearDerivative(double x)
        {
            return 1.0;
        }

        private static double gaussian(double x)
        {
            return Math.Exp(-Math.Pow(x, 2));
        }

        private static double gaussianDerivative(double x)
        {
            return -2.0 * x * gaussian(x);
        }

        private static double rationalSigmoid(double x)
        {
            return x / (1.0 + Math.Sqrt(1.0 + x * x));
        }

        private static double rationalSigmoidDerivative(double x)
        {
            double val = Math.Sqrt(1.0 + x * x);
            return 1.0 / (val * (1 + val));
        }
    }

    #endregion

    public class BackPropagation
    {

        #region Private data
        private int layerCount;
        private int inputSize;
        private int[] layerSize;
        private TransferFunction[] transferFunction;

        /* first index is layer, second index is node*/
        private double[][] layerOutput;
        private double[][] layerInput;
        private double[][] biases;
        private double[][] delta;
        private double[][] previousDelta;

        private double[][][] weight;
        private double[][][] previousWeight;

        #endregion

        #region Methods
        public void Run(ref double[] input, out double[] output)
        {
            // make sure we have enough data
            if(input.Length != inputSize)
            {
                throw new ArgumentException("input data not correct dimensions");
            }

            // Dimensions
            output = new double[layerSize[layerCount - 1]];

            /* Run Network */
            for(int l = 0; l < layerCount; l++)
            {
                for(int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                    {
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
                    }
                    sum += biases[l][j];
                    layerInput[l][j] = sum;

                    layerOutput[l][j] = TransferFunctions.Evaluate(transferFunction[l], sum);

                }
            }
            // copy output to output array
            for(int i = 0;i < layerSize[layerCount - 1]; i++)
            {
                output[i] = layerOutput[layerCount - 1][i];
            }
        }

        public double Train(ref double[] input, ref double[] desired, double TrainingRate, double Momentum)
        {
            // Parameter Validation
            if(input.Length != inputSize)
            {
                throw new ArgumentException("Invalid input parameter");
            }
            if(desired.Length != layerSize[layerCount - 1])
            {
                throw new ArgumentException("Invalid input parameter");
            }

            // local variables
            double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;
            double[] output = new double[layerSize[layerCount - 1]];

            // Run the Network
            Run(ref input, out output);
            // Back propagate the error
            for(int l = layerCount-1;l >= 0; l--)
            {
                // Output layer
                if(l == layerCount - 1)
                {
                    for (int k =0; k <layerSize[l]; k++)
                    {
                        delta[l][k] = output[k] - desired[k];
                        error += Math.Pow(delta[l][k], 2);
                        delta[l][k] *= TransferFunctions.EvaluateDerivative(transferFunction[l], layerInput[l][k]);
                    }
                }
                else // Hidden Layer
                {
                    for (int i = 0; i < layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for(int j = 0; j < layerSize[l+1]; j++)
                        {
                            sum += weight[l + 1][i][j] * delta[l+1][j];
                        }
                        sum *= TransferFunctions.EvaluateDerivative(transferFunction[l], layerInput[l][i]);

                        delta[l][i] = sum;
                    }
                }
            }
            // Update the weights and biases
            for(int l = 0; l < layerCount; l++)
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
                        weight[l][i][j] -= weightDelta + Momentum * previousWeight[l][i][j];

                        previousWeight[l][i][j] = weightDelta; 
                    }
                
            for(int l = 0; l < layerCount; l++)
                for(int i = 0; i < layerSize[l]; i++)
                {
                    biasDelta = TrainingRate * delta[l][i];
                    biases[l][i] -= biasDelta + Momentum * previousDelta[l][i];

                    previousDelta[l][i] = biasDelta;
                }
            return error;
        }
        
        public void Save(string FilePath)
        {
            if(FilePath == null)
            {
                return;
            }
            XmlWriter writer = XmlWriter.Create(FilePath);
            // Begin Document
            writer.WriteStartElement("NeuralNetwork");
            writer.WriteAttributeString("Type", "BackPropagation");

            writer.WriteStartElement("Parameters");

            writer.WriteElementString("Name", Name);
            writer.WriteElementString("inputSize", inputSize.ToString());
            writer.WriteElementString("layerCount", layerCount.ToString());

            writer.WriteStartElement("layers");

            for(int i = 0; i < layerCount; i++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", i.ToString());
                writer.WriteAttributeString("Size", layerSize[i].ToString());
                writer.WriteAttributeString("Type", transferFunction[i].ToString());

                writer.WriteEndElement(); // layer
            }

            writer.WriteEndElement(); // layers
            writer.WriteEndElement(); // Parameters

            // weights and biases
            writer.WriteStartElement("Weights");
            for(int i = 0;i < layerCount; i++)
            {
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Index", i.ToString());
                for(int j = 0; j < layerSize[i]; j++)
                {
                    writer.WriteStartElement("Node");

                    writer.WriteAttributeString("Index", j.ToString());
                    writer.WriteAttributeString("Bias", biases[i][j].ToString());

                    for(int l = 0; l < (l == 0 ? inputSize : layerSize[i - 1]); l++)
                    {
                        writer.WriteStartElement("Axon");

                        writer.WriteAttributeString("Index", l.ToString());
                        writer.WriteString(weight[i][j][l].ToString());

                        writer.WriteEndElement(); // Axon
                    }
                    writer.WriteEndElement(); // Node
                }

                writer.WriteEndElement(); // Layer
            }

            writer.WriteEndElement(); // Weights
            writer.WriteEndElement(); // NeuralNetwork

            writer.Flush();
            writer.Close();
        }
        #endregion

        #region Public Data
        public string Name = "Default";
        #endregion

        #region Constructors
        public BackPropagation(int[] layerSizes, TransferFunction[] transferFunctions)
        {
            // validate input
            if(transferFunctions.Length != layerSizes.Length || transferFunctions[0] != TransferFunction.None)
            {
                throw new ArgumentException("Cannot construct network");
            }

            // Initialize network layers
            layerCount = layerSizes.Length - 1;
            inputSize = layerSizes[0];
            layerSize = new int[layerCount];

            for(int i = 0; i < layerCount; i++)
            {
                layerSize[i] = layerSizes[i + 1];
            }

            transferFunction = new TransferFunction[layerCount];
            for(int i = 0;i < layerCount; i++)
            {
                transferFunction[i] = transferFunctions[i + 1];
            }

            // Start dimensioning
            biases = new double[layerCount][];
            previousDelta = new double[layerCount][];
            delta = new double[layerCount][];
            layerOutput = new double[layerCount][];
            layerInput = new double[layerCount][];

            weight = new double[layerCount][][];
            previousWeight = new double[layerCount][][];

            // fill second dimension
            for(int l = 0;l < layerCount; l++)
            {
                biases[l] = new double[layerSize[l]];
                previousDelta[l] = new double[layerSize[l]];
                delta[l] = new double[layerSize[l]];

                layerOutput[l] = new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];

                // Tertiary for case of first input needing the previous layer but does not exist
                weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
                previousWeight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    weight[l][i] = new double[layerSize[l]];
                    previousWeight[l][i] = new double[layerSize[l]];

                }
            }

            // Initialize the Weights
            for (int l = 0; l < layerCount; l++)
            {
                for(int j = 0; j < layerSize[l]; j++)
                {
                    biases[l][j] = Gausian.getRandomGaussian();
                    previousDelta[l][j] = 0.0;
                    layerOutput[l][j] = 0.0;
                    layerInput[l][j] = 0.0;
                    delta[l][j] = 0.0;
                }
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for(int j = 0; j < layerSize[l]; j++)
                    {
                        weight[l][i][j] = Gausian.getRandomGaussian();
                        previousWeight[l][i][j] = 0.0;
                    }
                }
            }

        }
        #endregion
    }

    public static class Gausian
    {
        private static Random gen = new Random();

        public static double getRandomGaussian()
        {
            return getRandomGaussian(0.0, 1.0);
        }
        public static double getRandomGaussian(double mean, double stdDev)
        {
            double rVal1, rVal2;

            getRandomGaussian(mean, stdDev, out rVal1, out rVal2);
            return rVal1;
        }
        public static void getRandomGaussian(double mean, double stdDev, out double val1, out double val2)
        {
            double u, v, s, t;

            do
            {
                u = 2 * gen.NextDouble() - 1;
                v = 2 * gen.NextDouble() - 1;
            } while (u * u + v * v > 1 || u == 0 && v == 0);

            s = u * u + v * v;
            t = Math.Sqrt(-2.0 * Math.Log(s)/s);

            val1 = stdDev * u * t + mean;
            val2 = stdDev * v * t + mean;
        }
    }
}
