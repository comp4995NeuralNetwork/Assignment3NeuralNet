using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeauralNetworkApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = new int[3] { 2, 2, 1 };
            TransferFunction[] tFuncs = new TransferFunction[3] { TransferFunction.None,
                                                                  TransferFunction.Sigmoid,
                                                                  TransferFunction.Linear };
            BackPropagation bpn = new BackPropagation(layerSizes, tFuncs);

            double[][] input, output;
            input = new double[4][]; output = new double[4][];
            for(int i = 0; i < 4; i++)
            {
                input[i] = new double[2]; output[i] = new double[1];
            }
            input[0][0] = 0.0; input[0][1] = 0.0; output[0][0] = 0.0; // Case 1 F xor F = F
            input[1][0] = 1.0; input[1][1] = 0.0; output[1][0] = 1.0; // Case 2 T xor F = T
            input[2][0] = 0.0; input[2][1] = 1.0; output[2][0] = 1.0; // Case 3 F xor T = T
            input[3][0] = 1.0; input[3][1] = 1.0; output[3][0] = 0.0; // Case 4 T xor T = F

            // Train
            double error = 0.0;
            int max_count = 10000, count = 0;

            do
            {
                // Prepare training epoch
                count++;
                error = 0.0;

                // Train
                for(int i = 0; i < 4; i++)
                {
                    // TrainRate and Momentm picked arbitrarilly
                    error += bpn.Train(ref input[i], ref output[i], 0.15, 0.10);
                }
                // Show Progress
                if (count % 100 == 0)
                {
                    Console.WriteLine("Epoch {0} completed with error {1:0.0000}", count, error);
                }
            } while (error > 0.0001 && count <= max_count);
            // Display results

            double[] networkOutput = new double[1];
            for (int i = 0; i < 4; i++)
            {
                bpn.Run(ref input[i], out networkOutput);
                Console.WriteLine("Case {3}: {0:0.0} xor {1:0.0} = {2:0.0000}", input[i][0], input[i][1], networkOutput[0], i+1);
            }
            
            // End Program
            Console.WriteLine("Press Enter ...");
            Console.ReadLine();
        }
    }
}
