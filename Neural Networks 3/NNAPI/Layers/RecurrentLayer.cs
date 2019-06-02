using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_3.NNAPI.Layers
{
    class RecurrentLayer : Layer
    {
        Dictionary<Neuron, Neuron> recurrentNeurons;
        string RecurrentActivationFunction;

        public RecurrentLayer(int numberOfNeurons, string ActivationFunction, string RecurrentActivationFunction, bool useBias = true)
        {
            neurons = new List<Neuron>();
            recurrentNeurons = new Dictionary<Neuron, Neuron>();
            this.ActivationFunction = ActivationFunction;
            this.RecurrentActivationFunction = RecurrentActivationFunction;
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(false, useBias));
                recurrentNeurons.Add(neurons[i], new Neuron(false, false));
            }

            ConnectRecurrentNeurons();
        }

        public void ConnectRecurrentNeurons()
        {
            for(int i = 0; i < recurrentNeurons.Count; i++)
            {
                recurrentNeurons[neurons[i]].ConnectTo(neurons[i]);
                neurons[i].ConnectTo(recurrentNeurons[neurons[i]]);
            }
        }

        public override void CalculateHiddenErrors(Layer NextLayer)
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.error = 0;
                foreach (Neuron nextNeuron in NextLayer.neurons)
                {
                    neuron.CalculateError(nextNeuron.error, neuron.connections[nextNeuron]);
                }
                neuron.CalculateError(recurrentNeurons[neuron].error, neuron.connections[recurrentNeurons[neuron]]);
                neuron.DerivativeActivation(ActivationFunction);

                recurrentNeurons[neuron].error = 0;
                recurrentNeurons[neuron].CalculateError(neuron.error, recurrentNeurons[neuron].connections[neuron]);
                recurrentNeurons[neuron].DerivativeActivation(RecurrentActivationFunction);
            }
        }

        public override void CalculateValues(Layer PreviousLayer)
        {
            foreach (Neuron neuron in neurons)
            {
                if (!neuron.isLocked)
                {
                    neuron.value = 0;
                    foreach (Neuron previousNeuron in PreviousLayer.neurons)
                    {
                        neuron.AddValue(previousNeuron.value, previousNeuron.connections[neuron]);
                    }
                    neuron.AddValue(recurrentNeurons[neuron].value, recurrentNeurons[neuron].connections[neuron]);
                    neuron.Activate(ActivationFunction);

                    recurrentNeurons[neuron].value = 0;
                    recurrentNeurons[neuron].AddValue(neuron.value, neuron.connections[recurrentNeurons[neuron]]);
                    recurrentNeurons[neuron].Activate(RecurrentActivationFunction);
                }
            }
        }

        public override void AdjustWeights(Layer NextLayer, double learningRate, bool useMomentum, double momentum)
        {
            foreach (Neuron neuron in neurons)
            {
                foreach (Neuron nextNeuron in NextLayer.neurons)
                {
                    neuron.AdjustWeights(nextNeuron, learningRate, useMomentum, momentum);
                }
                neuron.AdjustWeights(recurrentNeurons[neuron], learningRate, useMomentum, momentum);
                recurrentNeurons[neuron].AdjustWeights(neuron, learningRate, useMomentum, momentum);
            }
        }
    }
}
