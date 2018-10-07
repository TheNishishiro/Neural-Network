using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_2
{
    // relu

    class settings
    {
        public static NeuralNetworkv2 NN;
        //public static NeuralNetwork NN;
        private static double correctRate = 0, correctRatio = 0;
        public static double GlobalcorrectRate = 0, GlobalcorrectRatio = 0;
        public static int goal = 30000, c = 0;
        public static double LearningRate = 0.01;
        public static double Momentum = 0.9;
        public static int[] array = { 2, 5, 6, 8, 5, 7, 9, 5, 6, 2 };
        public static int numberOfLayers = array.Length;


        public static Texture2D SmallDot;

        private static Color color;
        private static double connectionWidthMult = 1;
        private static int maxBalls = 1000;
        public static Random rnd = new Random();
        public static List<Ball> balls = new List<Ball>();


        public static int function(double X, double Y)
        {
            if (Y < (Math.Sin(X / 100) * 200) + 350)
                return 0;
            else
                return 1;
        }

        public static void DrawNeuronsNew()
        {
            float heightAdjust = 0, heightAdjust2 = 0;
            for (int i = 0; i < NN.Layers.Count; i++)
            {
                for (int j = 0; j < NN.Layers[i].NumberOfNodes; j++)
                {
                    if (i == 0)
                        heightAdjust = (70 * (NN.Layers[1].NumberOfNodes / NN.Layers[0].NumberOfNodes)) - 70;
                    else if (i == NN.Layers.Count - 1)
                    {
                        heightAdjust = (70 * (NN.Layers[i - 1].NumberOfNodes / NN.Layers[i].NumberOfNodes)) - 70;
                    }
                    else if (i == NN.Layers.Count - 2)
                    {
                        heightAdjust2 = (70 * (NN.Layers[i].NumberOfNodes / NN.Layers[i+1].NumberOfNodes)) - 70;
                        heightAdjust = 0;
                    }
                    else
                    {
                        heightAdjust = 0;
                        heightAdjust2 = 0;
                    }

                    for (int k = 0; k < NN.Layers[i].NumberOfChildNodes; k++)
                    {
                        if (NN.Layers[i].Weights[j][k] < 0)
                            color = Color.Black;
                        else
                            color = Color.White;

                        if (Math.Abs(NN.Layers[i].Weights[j][k]) >= 1)
                            NFramework.NDrawing.Draw_Line_Between_Points(new Vector2((100 * i) + 64, (j * 70) + 64 + heightAdjust), new Vector2((115 + 100 * i) + 64, (k * 70) + 64 + heightAdjust2), "line", Math.Abs(NN.Layers[i].Weights[j][k]) * connectionWidthMult, color);
                        else if(Math.Abs(NN.Layers[i].Weights[j][k]) < 1 && Math.Abs(NN.Layers[i].Weights[j][k]) > 0.05)
                            NFramework.NDrawing.Draw_Line_Between_Points(new Vector2((100 * i) + 64, (j * 70) + 64 + heightAdjust), new Vector2((115 + 100 * i) + 64, (k * 70) + 64 + heightAdjust2), "line", 1, color);

                    }
                    NFramework.NDrawing.Draw("dot", new Vector2((100 * i)+64, (j * 70)+64 + heightAdjust), Color.White, 1, 0, true);
                    NFramework.NDrawing.DrawText(NN.Layers[i].NeuronValues[j].ToString("0.00"), new Vector2(((100 * i)-15)+64, ((j * 70)+64)-5 + heightAdjust), Color.Red);
                }
            }
        }


        public static void DrawSimulation(SpriteBatch spriteBatch)
        {
            for(int i = 0; i < balls.Count; i++)
            {
                NFramework.NDrawing.Draw("dot", balls[i].Position, balls[i].color, 1, 0, true);
            }
            for (float X = 0; X < 1300; X += 0.1f)
            {
                //(float)Math.Pow(Math.E, X/100)+120
                spriteBatch.Draw(SmallDot, new Vector2(X, (float)(Math.Sin(X/100)*200)+350), Color.White);
            }
        }

        private static void CalculateProc()
        {
            for(int i = 0; i < balls.Count; i++)
            {
                if (balls[i].correct == balls[i].guess)
                    correctRate++;
            }
            correctRatio = correctRate / balls.Count;
            correctRate = 0;

            if (balls.Count > maxBalls)
                balls.RemoveAt(0);
        }

        public static void DrawInfo()
        {
            NFramework.NDrawing.DrawText("Current %:           " + correctRatio.ToString("0.0000"), new Vector2(900, 0), Color.Red);
            NFramework.NDrawing.DrawText("Overall % :           " + GlobalcorrectRatio.ToString("0.0000"), new Vector2(900, 20), Color.Red);
            NFramework.NDrawing.DrawText("Number of tests: " + c.ToString(), new Vector2(900, 40), Color.Red);

            CalculateProc();
        }
    }
}
