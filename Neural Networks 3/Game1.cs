using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Neural_Networks_3.NNAPI;
using System;
using static Neural_Networks_2.settings;

namespace Neural_Networks_2
{
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        NeuralNetwork model = new NeuralNetwork(0.01, true, 0.8);

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

            model.AddDense(2, "relu", false);
            model.AddRecurrent(5, "relu", "relu");
            model.AddRecurrent(6, "relu", "relu");
            model.AddRecurrent(8, "relu", "relu");
            model.AddRecurrent(5, "relu", "relu");
            model.AddDense(2, "sigmoid");

            model.Compile();
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
                double[] inputs = { x/1300, y/700 };
                model.SetInputs(inputs);

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

                model.SetDesiredOutput(output);
                model.FeedForward();

                if (nodeActive == model.GetMaxOutputID())
                   GlobalcorrectRate++;

                GlobalcorrectRatio = GlobalcorrectRate / c;
                model.Backpropagate();
                balls.Add(new Ball(x, y, model.GetMaxOutputID(), (int)nodeActive));

                //if (Keyboard.GetState().IsKeyDown(Keys.A))
                //    DrawNeuronsNew();
                //else
                    DrawSimulation(spriteBatch);
                    
                DrawInfo();
                c++;
            }

            

            spriteBatch.End();
            base.Draw(gameTime);
        }
    }
}
