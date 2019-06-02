using Neural_Networks_3.NNAPI.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_3.NNAPI
{
    class NeuralNetwork
    {
        List<Layer> NNLayers;
        double LearningRate, MomentumFactor;
        bool useMomentum;

        public NeuralNetwork(double LearningRate, bool useMomentum, double MomentumFactor)
        {
            NNLayers = new List<Layer>();
            this.LearningRate = LearningRate;
            this.useMomentum = useMomentum;
            this.MomentumFactor = MomentumFactor;
        }

        public void AddDense(int NeuronCount, string activation, bool useBias = true)
        {
                NNLayers.Add(new DenseLayer(NeuronCount, activation, useBias));
        }

        public void AddRecurrent(int NeuronCount, string activation, string recurrentActivation, bool useBias = true)
        {
            NNLayers.Add(new RecurrentLayer(NeuronCount, activation, recurrentActivation, useBias));
        }

        public void Fit(double[][] InputArray, double[][] ExpectedValues, int Epochs)
        {
            double error = 0;
            for (int eps = 0; eps < Epochs; eps++)
            {
                for (int i = 0; i < InputArray.GetLength(0); i++)
                {
                    SetInputs(InputArray[i]);
                    SetDesiredOutput(ExpectedValues[i]);
                    FeedForward();
                    Backpropagate();

                    error = CalculateError();
                }
                Console.WriteLine($"[{eps}/{Epochs}]: Error: {error}");
            }
        }


        public double CalculateError()
        {
            double error = 0;
            double[] output = GetOutput();
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Abs(output[i] - NNLayers[NNLayers.Count - 1].neurons[i].desiredValue);
            }
            return error / output.Length;
        }

        public int GetMaxOutputID()
        {
            double[] outputs = GetOutput();
            double max = double.MaxValue * -1;
            int id = 0;

            for(int i = 0; i < outputs.Length; i++)
            {
                if(max < outputs[i])
                {
                    id = i;
                    max = outputs[i];
                }
            }

            return id;
        }

        public void SetInputs(double[] InputArray)
        {
            for(int i = 0; i < InputArray.Length; i++)
            {
                NNLayers[0].neurons[i].value = InputArray[i];
            }
        }

        public void SetDesiredOutput(double[] OutputArray)
        {
            for (int i = 0; i < OutputArray.Length; i++)
            {
                NNLayers[NNLayers.Count-1].neurons[i].desiredValue = OutputArray[i];
            }
        }

        public void Backpropagate()
        {
            NNLayers[NNLayers.Count - 1].CalculateOutputErrors();
            for (int i = NNLayers.Count - 2; i >= 0; i--)
            {
                NNLayers[i].CalculateHiddenErrors(NNLayers[i + 1]);
            }
            for (int i = NNLayers.Count - 2; i >= 0; i--)
            {
                NNLayers[i].AdjustWeights(NNLayers[i + 1], LearningRate, useMomentum, MomentumFactor);
            }
        }

        public void FeedForward()
        {
            for (int i = 1; i < NNLayers.Count; i++)
            {
                NNLayers[i].CalculateValues(NNLayers[i - 1]);
            }
        }

        public double[] GetOutput()
        {
            double[] values = new double[NNLayers[NNLayers.Count - 1].neurons.Count];
            for(int i = 0; i < NNLayers[NNLayers.Count - 1].neurons.Count; i++)
            {
                values[i] = NNLayers[NNLayers.Count - 1].neurons[i].value;
            }
            return values;
        }

        public void Compile()
        {
            for(int i = 1; i < NNLayers.Count; i++)
            {
                NNLayers[i].FullyConnect(NNLayers[i - 1]);
            }
        }
    }
}
