using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_3.NNAPI.Layers
{
    class DenseLayer : Layer
    {
        public DenseLayer(int numberOfNeurons, string ActivationFunction, bool useBias = true)
        {
            neurons = new List<Neuron>();
            this.ActivationFunction = ActivationFunction;

            for (int i = 0; i < numberOfNeurons; i++)
                neurons.Add(new Neuron(false, useBias));
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
                neuron.DerivativeActivation(ActivationFunction);
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
                    neuron.Activate(ActivationFunction);
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
            }
        }
    }
}
