using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_3.NNAPI.Layers
{
    abstract class Layer
    {
        public List<Neuron> neurons;
        public string ActivationFunction;

        public abstract void CalculateHiddenErrors(Layer NextLayer);
        public abstract void CalculateValues(Layer PreviousLayer);
        public abstract void AdjustWeights(Layer NextLayer, double learningRate, bool useMomentum, double momentum);

        public void CalculateOutputErrors()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.error = (neuron.desiredValue - neuron.value);
                neuron.DerivativeActivation(ActivationFunction);
            }
        }

        public void FullyConnect(Layer PreviousLayer)
        {
            int neuronCount = PreviousLayer.neurons.Count;

            foreach (Neuron mainNeuron in neurons)
            {
                for (int i = 0; i < neuronCount; i++)
                {
                    if (!PreviousLayer.neurons[i].isLocked)
                    {
                        PreviousLayer.neurons[i].ConnectTo(mainNeuron);
                    }
                }
            }
        }
    }
}
