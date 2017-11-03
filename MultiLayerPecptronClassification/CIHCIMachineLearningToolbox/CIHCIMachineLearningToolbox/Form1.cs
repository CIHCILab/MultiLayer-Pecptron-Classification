using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MachineLearning;

namespace CIHCIMachineLearningToolbox
{



    
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MultiLayerPerceptron MLP = new MultiLayerPerceptron("579.txt", "2,3,4", 1000, 0.1);

            MLP.RUN();

            string s = "train rmse:"+MLP.trainrmse.ToString() + "\n" + "train correct:" + MLP.traincorrect.ToString() + "%\n" + "test rmse:" + MLP.testrmse.ToString() + "\n" + "test correct:" + MLP.testcorrect.ToString() + "%\n";

            listBox1.Items.Add("train rmse:" + MLP.trainrmse.ToString());
            listBox1.Items.Add("train correct:" + MLP.traincorrect.ToString());
            listBox1.Items.Add("test rmse:" + MLP.testrmse.ToString());
            listBox1.Items.Add("test correct:" + MLP.testcorrect.ToString());
        }
    }
}
