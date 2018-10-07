using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;
using static Neural_Networks_2.settings;

namespace Neural_Networks_2
{
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
           //TargetElapsedTime = TimeSpan.FromSeconds(1.0f / 1000.0f);

            graphics.SynchronizeWithVerticalRetrace = false;
            IsFixedTimeStep = false;

            graphics.PreferredBackBufferHeight = 700;
            graphics.PreferredBackBufferWidth = 1300;
        }

        protected override void Initialize()
        {
           // IsFixedTimeStep = false;
            Form1 f1 = new Form1();
            //f1.Show();
            base.Initialize();
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            NFramework.NDrawing.SetFramework(spriteBatch, Content);
            SmallDot = Content.Load<Texture2D>("small");

            NFramework.NDrawing.AddTexture("dotw", "dot");
            NFramework.NDrawing.AddFont("font", "font");
            NFramework.NDrawing.AddPixelTexture("line", GraphicsDevice);

            //NN = new NeuralNetwork(2, 5, 2);

            NN = new NeuralNetworkv2(numberOfLayers, array);
            NN.SetLearningRate(LearningRate);
            NN.SetLinearOutput(false);
            NN.SetMomentum(true, Momentum);

        }

        protected override void Update(GameTime gameTime)
        {
           
            
            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);
            spriteBatch.Begin();

            if (c > -1)
            {
                double x = 0, y, error = 0, nodeActive = 0;
                double[] output = { 0, 0 };

                x = rnd.Next(graphics.PreferredBackBufferWidth);
                y = rnd.Next(graphics.PreferredBackBufferHeight);
                NN.SetInput(0, x / 1300);
                NN.SetInput(1, y / 700);

                if (function(x,y) == 0)//(y < 350)
                {
                    output[0] = 1;
                    nodeActive = 0;
                }
                else
                {
                    output[1] = 1;
                    nodeActive = 1;
                }

                for(int i = 0; i < array[array.Length-1]; i++)
                {
                    if(i < array.Length-1)
                        NN.SetDesiredOutput(i, output[i]);
                }

                
                //NN.SetDesiredOutput(1, output[1]);

                NN.FeedForward();
                error += NN.CalculateError();
                if (nodeActive == NN.GetMaxOutputID())
                   GlobalcorrectRate++;

                GlobalcorrectRatio = GlobalcorrectRate / c;
                NN.BackPropagate();
                balls.Add(new Ball(x, y, NN.GetMaxOutputID(), (int)nodeActive));

                if (Keyboard.GetState().IsKeyDown(Keys.A))
                    DrawNeuronsNew();
                else
                    DrawSimulation(spriteBatch);
                    
                DrawInfo();
                c++;
            }

            

            spriteBatch.End();
            base.Draw(gameTime);
        }
    }
}
