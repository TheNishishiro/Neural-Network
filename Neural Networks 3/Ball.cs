using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks_2
{
    class Ball
    {
        public Vector2 Position;
        public int guess;
        public int correct;
        public Color color;

        public Ball(double x, double y, int guess, int correct)
        {
            this.guess = guess;
            this.correct = correct;

            Position = new Vector2((float)x, (float)y);
            if (guess == 0)
                color = Color.White;
            else
                color = Color.Black;
        }
    }
}
