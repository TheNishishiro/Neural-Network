# About

A project to implement neural networks from scratch in C# 

## Introduction

For now the code is very messy as it's overcomming and overhaul from previous iteration, new code and architectrue can be found in *NNAPI* folder.

Everything outside that folder is either garbage or used to visualize progress of learning.

Current implementation only consists of dense MLP layers but the whole point of the overaul was to bring more different layers into play as well as add much more low-level customization so it's still very early into development. (considering I haven't worked on it since about 2 years).

For now I mostly tested it for binary classification and it reaches accuraccy of around 98% so similiar to the old code which if I remember correctly did a little better but at least it's new and mine.
I didn't test it on any more complex tasks and it probably wouldn't perform all that well either.

## Geting started

Follow intructions below to run/use this project in your environment.

### Prerequisites

You'll need to install following if you want to run the simulation, if you only want to use my nerual network then skip it
```
Monogame
```
### Usage

Just copy everything from **NNAPI** folder, it contains everything regarding neural network.
Then to create model you will want to call for 
`NeuralNetwork model = new NeuralNetwork([learning Rate], [useMomentum], [Momentum factor]);`
then just add layers via
```
model.AddDense([neurons], [activation function], [use bias]);
model.AddDense(5, "relu");
model.AddDense(2, "sigmoid");
```
finally aftyer you are done adding layers call `model.Compile()` which will connect all the layers together and stuff.

Remember to mark first later to not use bias (will probably be changed later on)

then either call `model.fit` or use separate functions to your liking like this:
```
SetInputs([InputArray]);
SetDesiredOutput([ExpectedValues]);
FeedForward();
Backpropagate();
```

and to finally get values from model you can call `model.GetOutput()`
That's pretty much it

## To do

- [x] Dense MLP layers
- [ ] Recurrent layers
- [ ] Convolutional layers 

## License

You can use my code with a bit of credit (idk why you would tho).
