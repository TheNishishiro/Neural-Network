using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Neural_Networks_2.settings;

namespace Neural_Networks_2
{
    class NeuralNetworkLayer
    {
        private string activ = "relu"; // sigmoid, tanh, relu

       public int NumberOfNodes;
       public int NumberOfChildNodes;
       public int NumberOfParentNodes;
       public double[][] Weights;
       public double[][] WeightChanges;
       public double[] NeuronValues;
       public double[] DesiredValues;
       public double[] Errors;
       public double[] BiasWeights;
       public double[] BiasValues;
       public double LearningRate;
       public bool LinearOutput;
       public bool UseMomentum;
       public double MomentumFactor;
       public NeuralNetworkLayer ParentLayer;
       public NeuralNetworkLayer ChildLayer;

        public NeuralNetworkLayer()
        {
            ParentLayer = null;
            ChildLayer = null;
            LinearOutput = false;
            UseMomentum = false;
            MomentumFactor = 0.9;
        }

        public void Initialize(int NumNodes, NeuralNetworkLayer parent, NeuralNetworkLayer child)
        {
            int i, j;

            NeuronValues = new double[NumberOfNodes];
            DesiredValues = new double[NumberOfNodes];
            Errors = new double[NumberOfNodes];

            if(parent != null)
            {
                ParentLayer = parent;
            }
            if(child != null)
            {
                ChildLayer = child;
                Weights = new double[NumberOfNodes][];
                WeightChanges = new double[NumberOfNodes][];
                for(i = 0; i < NumberOfNodes; i++)
                {
                    Weights[i] = new double[NumberOfChildNodes];
                    WeightChanges[i] = new double[NumberOfChildNodes];
                }
                BiasValues = new double[NumberOfChildNodes];
                BiasWeights = new double[NumberOfChildNodes];
            }
            else
            {
                Weights = null;
                BiasValues = null;
                BiasWeights = null;
                WeightChanges = null;
            }
            for (i = 0; i < NumberOfNodes; i++)
            {
                NeuronValues[i] = 0;
                DesiredValues[i] = 0;
                Errors[i] = 0;

                if (ChildLayer != null)
                    for (j = 0; j < NumberOfChildNodes; j++)
                    {
                        Weights[i][j] = 0;
                        WeightChanges[i][j] = 0;
                    }
            }
            if (ChildLayer != null)
                for (j = 0; j < NumberOfChildNodes; j++)
                {
                    BiasValues[j] = 1;
                    BiasWeights[j] = 0;
                }

        }
        public void RandomizeWeights()
        {
            int i, j;
            int min = 0;
            int max = 200;
            int number;
            for (i = 0; i < NumberOfNodes; i++)
            {
                for (j = 0; j < NumberOfChildNodes; j++)
                {
                    number = rnd.Next(min, max);
                    if (number > max)
                        number = max;
                    if (number < min)
                        number = min;
                    Weights[i][j] = number / 100.0f - 1;
                }
            }

            for (j = 0; j < NumberOfChildNodes; j++)

            {

                number = rnd.Next(min, max);

                if (number > max)

                    number = max;

                if (number < min)

                    number = min;

                BiasWeights[j] = number / 100.0f - 1;

            }



        }
        public void CalculateNeuronValues()
        {
            int i, j;
            double x;
            if (ParentLayer != null)
            {
                for (j = 0; j < NumberOfNodes; j++)
                {
                    x = 0;
                    for (i = 0; i < NumberOfParentNodes; i++)
                    {
                        x += ParentLayer.NeuronValues[i] *ParentLayer.Weights[i][j];
                    }
                    x += ParentLayer.BiasValues[j] *ParentLayer.BiasWeights[j];
                    if ((ChildLayer == null) && LinearOutput)
                    {
                        NeuronValues[j] = x;
                    }
                    else
                    {
                        NeuronValues[j] = ActivationFunction(activ, x);
                    }

                }
            }
        }

        private double ActivationFunction(string activation, double x)
        {
            double value = 0;

            if (activation == "sigmoid")
                value = 1.0f / (1 + Math.Exp(-x));
            if (activation == "tanh")
                value = Math.Tanh(x);
            if (activation == "relu")
                value = Math.Max(0.01 * x, x);

            return value;
        }

        private double DerivativeActivationFunction(string activation, int i)
        {
            double value = 0;

            if (activation == "sigmoid")
                value = NeuronValues[i] * (1.0f - NeuronValues[i]);
            if (activation == "tanh")
                value = 1 - Math.Pow(NeuronValues[i], 2);
            if (activation == "relu")
            {
                if (NeuronValues[i] > 0)
                    value = 1;
                else
                    value = 0.01;
            }

            return value;
        }

        public void CalculateErrors()
        {
            int i, j;
            double sum;
            double derivative = 0.01;
            if (ChildLayer == null) // output layer
            {
                for (i = 0; i < NumberOfNodes; i++)
                {
                    derivative = DerivativeActivationFunction(activ, i);
                    Errors[i] = (DesiredValues[i] - NeuronValues[i]) * derivative;
                }
            }
            else if (ParentLayer == null)
            { // input layer
                for (i = 0; i < NumberOfNodes; i++)
                {
                    Errors[i] = 0.0f;
                }
            }
            else
            { // hidden layer
                for (i = 0; i < NumberOfNodes; i++)
                {
                    sum = 0;
                    for (j = 0; j < NumberOfChildNodes; j++)
                    {
                        sum += ChildLayer.Errors[j] * Weights[i][j];
                    }
                    derivative = DerivativeActivationFunction(activ, i);
                    Errors[i] = sum * derivative;
                }
            }

        }

        public void AdjustWeights()
        {
            int i, j;
            double dw;
            
            if (ChildLayer != null)
            {
                for (i = 0; i < NumberOfNodes; i++)
                {
                    for (j = 0; j < NumberOfChildNodes; j++)
                    {
                        dw = LearningRate * ChildLayer.Errors[j] *
                              NeuronValues[i];
                        if (UseMomentum)
                        {
                            Weights[i][j] += dw + MomentumFactor *
                                                   WeightChanges[i][j];
                            WeightChanges[i][j] = dw;
                        }
                        else
                        {
                            Weights[i][j] += dw;
                        }
                    }
                }

                for (j = 0; j < NumberOfChildNodes; j++)
                {
                    BiasWeights[j] += LearningRate *
                                           ChildLayer.Errors[j] *
                                           BiasValues[j];
                }
            }
        }
    }
}
