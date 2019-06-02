using Neural_Networks_2;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_3.NNAPI
{
    class Neuron
    {
        public double biasValue;
        public double biasWeight;
        public double value;
        public double desiredValue;
        public double error;
        public Dictionary<Neuron, double> connections;
        public Dictionary<Neuron, double> connectionsChanges;
        public bool useBias;
        public bool isLocked;
        public bool beenBackpropagated;

        public Neuron(bool isLocked = false, bool useBias = false)
        {
            value = (settings.rnd.NextDouble() * 2) - 1;
            biasValue = 1;
            biasWeight = (settings.rnd.NextDouble() * 2) - 1;

            desiredValue = 0;
            error = 0;
            this.useBias = useBias;
            this.isLocked = isLocked;
            beenBackpropagated = false;
            connections = new Dictionary<Neuron, double>();
            connectionsChanges = new Dictionary<Neuron, double>();
        }

        public void AddValue(double parentValue, double weight)
        {
            value += parentValue * weight;
        }

        public void CalculateError(double childError, double weight)
        {
            error += childError * weight;
        }

        public void AdjustWeights(Neuron child, double learningRate, bool useMomentum, double momentum)
        {
            double dw = learningRate * child.error * value;
            if (useMomentum)
            {
                dw += momentum * connectionsChanges[child];
                connectionsChanges[child] = dw;
            }
            connections[child] += dw;

            if(useBias)
                biasWeight += learningRate * child.error * biasValue;
        }

        public void Activate(string activation)
        {
            if (useBias)
                value += biasValue * biasWeight;

            switch (activation)
            {
                case "sigmoid":
                    value = 1.0f / (1 + Math.Exp(-value));
                    break;
                case "tanh":
                    value = Math.Tanh(value);
                    break;
                case "relu":
                    value = Math.Max(0.01 * value, value);
                    break;
                case "linear":
                    value = value;
                    break;
            }
        }

        public void DerivativeActivation(string activation)
        {
            switch(activation)
            {
                case "sigmoid":
                    error *= value * (1.0f - value);
                    break;
                case "tanh":
                    error *= 1 - Math.Pow(value, 2);
                    break;
                case "relu":
                    if (value > 0)
                        error *= 1;
                    else
                        error *= 0.01;
                    break;
                default:
                    error = error;
                    break;

            }
        }

        public void LockNeuron() => isLocked = true;
        public void ConnectTo(Neuron neuron)
        {
            connections.Add(neuron, (settings.rnd.NextDouble() * 2) - 1);
            connectionsChanges.Add(neuron ,0);
        }

    }
}
