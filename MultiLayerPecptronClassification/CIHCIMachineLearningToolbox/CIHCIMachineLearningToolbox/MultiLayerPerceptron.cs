using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CIHCIMachineLearningToolbox
{

    class MultiLayerPerceptron
    {

        double[][] data;
        double[][] sorted_data;
        //List<double[][]> data = new List<double[][]>();
        //List<double[][]> sorted_data = new List<double[][]>();

        int n = 0;
        int m = 0;
        int[] category = new int[10];
        //List<int[]> category = new List<int[]>();
        int category_n = 1;
        double[][][] hidden_w;
        //List<double[][][]> hidden_w = new List<double[][][]>();
        double[][] out_w;
        //List<double[][]> out_w = new List<double[][]>();
        double[][] v;
        //List<double[][]> v = new List<double[][]>();
        double[] out_v;
        //List<double[]> out_v = new List<double[]>();
        int[] hlayer; // define how many nodes in each hidden layer
        //List<int[]> hlayer = new List<int[]>();
        int hlayer_n;
        double[][] h_grad; // hidden layers' gradient
        //List<double[][]> h_grad = new List<double[][]>();
        double[] o_grad;  // output layer's gradient
        //List<double[]> o_grad = new List<double[]>();
        double in_rmse = 0.0001;
        double learning_rate = 0.5;
        //TColor color_table[10] = {(TColor)0x00FF0000, (TColor)0x0000FF00, (TColor)0x000000FF, (TColor)0x00FFFF00, (TColor)0x00FF00FF,
        //                  (TColor)0x0000FFFF, (TColor)0x00880000, (TColor)0x00008800, (TColor)0x00000088, (TColor)0x00888800};
        int color = 0;
        int iteration = 10000;
        int IS_FILEOPEN = 0;
        int test_n;
        int train_n;
        Random rnd = new Random();
        //TStringList* sList1 = new TStringList;
        //TStringList* sList2 = new TStringList; // Use for dismember a string into numbers

        int dataNum = 0; // 資料筆數
        int dataDegree; // 資料維度
        int trainNum; // 訓練筆數
        int testNum; // 測試筆數
        int categoryNum = 0; // 資料群數

        double[][] totalData;
        int[] expect;
        int[] trainExpect;
        int[] testExpect;
        double[][] trainData;
        double[][] testData;
        double[] boundary;
        string listhidden;
        string filename;

        public double trainrmse = 0;
        public double testrmse = 0;
        public double traincorrect = 0;
        public double testcorrect = 0;
        public MultiLayerPerceptron(string file, string listhidden, int iteration, double learningrate)
        {
            this.filename = file;
            this.listhidden = listhidden;
            this.iteration = iteration;
            this.learning_rate = learningrate;
        }

        public void RUN()
        {
            Open();
            Train();
            Test();

        }


        void InitialNetwork()
        {
            // Clear Chart2
            //for(int i = 0; i <= category_n; i++)
            // Form1->Chart2->Series[i]->Clear();
            //Form1->Series9->Clear();

            //sList2->Delimiter = ',';
            //sList2->DelimitedText = Form1->EditHLayer->Text;
            //hlayer_n = sList2->Count;

            // Allocate memory to *hlayer & ***hidden_w & **out_w
            string[] eachData = listhidden.Split(new Char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            hlayer_n = eachData.Length;
            hlayer = new int[hlayer_n];
            hidden_w = new double[hlayer_n][][];
            out_w = new double[category_n][];

            for (int i = 0; i < hlayer_n; i++)
            {
                hlayer[i] = int.Parse(eachData[i]);// = sList2->Strings[i].ToInt();

                hidden_w[i] = new double[hlayer[i]][];

                if (i > 0)
                    for (int j = 0; j < hlayer[i]; j++)
                        hidden_w[i][j] = new double[hlayer[i - 1] + 1];
                else
                    for (int j = 0; j < hlayer[i]; j++)
                        hidden_w[0][j] = new double[m - 1];
            }

            for (int i = 0; i < category_n; i++)
                out_w[i] = new double[hlayer[hlayer_n - 1] + 1];

            // Initialize out_w[][]
            for (int i = 0; i < category_n; i++)
                for (int j = 0; j < hlayer[hlayer_n - 1] + 1; j++)
                    out_w[i][j] = rnd.Next(0, 2000) / 1000.0 - 1;



            // Initialize hidden_w[0][][]
            for (int i = 0; i < hlayer[0]; i++)
                for (int j = 0; j < m - 1; j++)
                    hidden_w[0][i][j] = rnd.Next(0, 2000) / 1000.0 - 1;


            // Initialize hidden_w[1~hlayer_n][][]
            for (int i = 1; i < hlayer_n; i++)
                for (int j = 0; j < hlayer[i]; j++)
                    for (int k = 0; k < hlayer[i - 1] + 1; k++)
                        hidden_w[i][j][k] = rnd.Next(0, 2000) / 1000.0 - 1;


            // Allocate memory to **v & **h_grad & *o_grad
            v = new double[hlayer_n][];
            h_grad = new double[hlayer_n][];
            o_grad = new double[category_n];

            for (int i = 0; i < hlayer_n; i++)
            {
                v[i] = new double[hlayer[i] + 1];
                h_grad[i] = new double[hlayer[i]];
            }

            // Initialize v[][]
            for (int i = 0; i < hlayer_n; i++)
                v[i][0] = -1;

            // Allocate memory to *out_v
            out_v = new double[category_n];

            this.learning_rate = this.learning_rate;//= StrToFloat(Form1->EditLearningRate->Text);
            this.iteration = this.iteration;//= StrToInt(Form1->EditIteration->Text);

            //if (Form1->CheckBox1->Checked)
            //    in_rmse = StrToFloat(Form1->EditErrorRate->Text);

        }


        void random_data()
        {
            double[] temp;

            int r1;
            int r2;
            category_n = 1;
            // Random the data[]

            for (int i = 0; i < n; i++)
            {
                //Memo1->Lines->Add(r);
                r1 = rnd.Next(0, n);
                r2 = rnd.Next(0, n);
                temp = data[r1];
                data[r1] = data[r2];
                data[r2] = temp;
            }

            // Bubble Sort
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n - 1; j++)
                    if (sorted_data[j][m - 1] > sorted_data[j + 1][m - 1])
                    {
                        temp = sorted_data[j + 1];
                        sorted_data[j + 1] = sorted_data[j];
                        sorted_data[j] = temp;
                    }

            // Find category_n
            for (int i = 0; i < n; i++)
                if (i != 0 && sorted_data[i - 1][m - 1] != sorted_data[i][m - 1])
                    category_n++;




            //Form1->LabelCategory_n->Caption = "群數       " + AnsiString(category_n);
            //Form1->LabelTrainNum->Caption = "訓練筆數      " + AnsiString(train_n);
            //Form1->LabelTestNum->Caption = "測試筆數      " + AnsiString(test_n);
        }
        //------------------------------------------------------------------------


        void Open()
        {
            //Microsoft.Win32.OpenFileDialog File = new Microsoft.Win32.OpenFileDialog();

            //if (File.ShowDialog() != true) return;


            string line;
            string[] eachData;

            bool[] flag;
            //Random rnd = new Random();


            //label6.Text = "資料檔案:        " + eachData[eachData.Length - 1];


            /*
            StreamReader strReader;

            using (FileStream fs = new FileStream("579.txt", FileMode.Open))
            {
                using (strReader = new StreamReader(fs))
                {

                    while (strReader.ReadLine() != null)
                    {
                        dataNum++;
                    }
                }
            }
            */

            StreamReader strReader = new StreamReader(filename);



            line = strReader.ReadLine();
            eachData = line.Split(new Char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (eachData.Length == 1)
            {

                eachData = line.Split(new Char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
            }

            dataDegree = eachData.Length - 1;

            IS_FILEOPEN = 1;

            strReader = new StreamReader(filename);

            dataNum = 0;

            while (strReader.ReadLine() != null)
            {
                dataNum++;
            }

            //eachData = File.FileName.Split('\\');





            double[][] AllData = new double[dataNum][];
            for (int i = 0; i < dataNum; i++)
                AllData[i] = new double[dataDegree + 1];

            totalData = new double[dataNum][];
            expect = new int[dataNum];

            for (int i = 0; i < dataNum; i++)
                totalData[i] = new double[dataDegree];

            strReader = new StreamReader(filename);

            line = strReader.ReadLine();
            if (line != null)
                eachData = line.Split(new Char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (eachData.Length == 1)
            {

                eachData = line.Split(new Char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
            }

            for (int i = 0; i < dataNum; i++)
            {

                //eachData = line.Split(new Char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                //eachData = line.Split(new Char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);

                for (int j = 0; j < dataDegree; j++)
                {
                    totalData[i][j] = double.Parse(eachData[j]);
                    //MessageBox.Show(i.ToString() + ", " + totalData[i][j].ToString());
                    AllData[i][j] = totalData[i][j];
                }


                expect[i] = (int)double.Parse(eachData[eachData.Length - 1]);
                AllData[i][dataDegree] = expect[i];

                //MessageBox.Show(expect[i].ToString());

                line = strReader.ReadLine();

                if (line != null)
                    eachData = line.Split(new Char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (eachData.Length == 1)
                {

                    eachData = line.Split(new Char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                }
            }

            int mmax = -99999;
            int mmin = 99999;
            for (int i = 0; i < dataNum; i++)
            {
                if (expect[i] > mmax) mmax = expect[i];
                if (expect[i] < mmin) mmin = expect[i];
            }
            if (mmin == 1)
            {
                categoryNum = mmax;
                for (int i = 0; i < dataNum; i++) expect[i] = expect[i] - 1;

            }
            else
                categoryNum = mmax + 1;



            testNum = dataNum / 2;
            trainNum = dataNum - testNum;

            flag = new bool[dataNum];

            for (int i = 0; i < trainNum; i++)
                flag[i] = false;

            // Allocate memory to trainData[][]
            trainData = new double[trainNum][];
            trainExpect = new int[trainNum];

            for (int i = 0; i < trainNum; i++)
                trainData[i] = new double[dataDegree];


            // 從資料集中隨機挑 trainNum 個當訓練集

            for (int i = 0; i < trainNum; i++)
            {

                int k = rnd.Next(0, dataNum);

                while (flag[k])
                {
                    k = rnd.Next(0, dataNum);
                }

                flag[k] = true;

                trainExpect[i] = expect[k];

                for (int j = 0; j < dataDegree; j++)
                    trainData[i][j] = totalData[k][j];

            }

            // Allocate memory to testData[][]
            testData = new double[testNum][];
            testExpect = new int[testNum];

            for (int i = 0; i < testNum; i++)
                testData[i] = new double[dataDegree];

            // 資料集中剩下的當作測試集
            for (int i = 0, k = 0; i < dataNum; i++)
            {

                if (flag[i])
                    continue; // 該筆已被當作訓練資料, 跳下一筆

                testExpect[k] = expect[i];

                for (int j = 0; j < dataDegree; j++)
                    testData[k][j] = totalData[i][j];

                k++;
            }

            this.n = dataNum;

            this.test_n = this.n / 2;
            this.train_n = this.n - test_n;

            this.data = new double[this.n][];
            this.sorted_data = new double[this.n][];
            this.m = eachData.Length + 1;

            for (int i = 0; i < n; i++)
            {
                this.data[i] = new double[m];
                this.sorted_data[i] = new double[m];
            }



            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m - 1; j++)
                {
                    data[i][0] = -1;
                    data[i][j + 1] = AllData[i][j];

                    sorted_data[i][0] = data[i][0];
                    sorted_data[i][j + 1] = data[i][j + 1];
                }
            }

            // Random the data[]
            for (int i = 0; i < this.n; i++)
            {
                //Memo1->Lines->Add(r);
                int r1 = rnd.Next(this.n);
                int r2 = rnd.Next(this.n);
                double[] temp = data[r1];
                data[r1] = data[r2];
                data[r2] = temp;
            }

            // Bubble Sort
            for (int i = 0; i < this.n; i++)
                for (int j = 0; j < this.n - 1; j++)
                    if (sorted_data[j][this.m - 1] > sorted_data[j + 1][this.m - 1])
                    {

                        double[] temp = sorted_data[j + 1];
                        sorted_data[j + 1] = sorted_data[j];
                        sorted_data[j] = temp;
                    }
            this.category_n = 1;
            // Find category_n
            for (int i = 0; i < this.n; i++)
                if (i != 0 && sorted_data[i - 1][this.m - 1] != sorted_data[i][this.m - 1])
                    this.category_n++;
        }



        void Train()
        {

            random_data();

            int E;
            int kind;
            int it = 0;
            double out_rmse;

            if (IS_FILEOPEN == 0)
            {
                //ShowMessage("尚未開啟檔案!");
                return;
            }

            InitialNetwork();

            // Add information to Memo1
            //Memo1->Lines->Add("Initial Weighted Vectors:");
            //PrintWeightedVectors();


            // Start Traning
            for (; it < iteration; it++)
            {
                for (int count = 0; count < train_n; count++)
                {
                    // Evaluate data[count][]
                    Evaluate(count);

                    E = (int)data[count][m - 1] - (int)sorted_data[0][m - 1];

                    // Backpropagation: Update w[][][] & gradient[]
                    // Evaluate o_grad
                    for (int i = 0; i < category_n; i++)
                    {
                        if (i == E)
                            o_grad[i] = (1 - out_v[i]) * out_v[i] * (1 - out_v[i]);
                        else
                            o_grad[i] = (0 - out_v[i]) * out_v[i] * (1 - out_v[i]);

                        for (int j = 0; j < hlayer[hlayer_n - 1] + 1; j++)
                            out_w[i][j] += learning_rate * o_grad[i] * v[hlayer_n - 1][j];

                    }// end for

                    //ShowMessage(AnsiString(count) + "After Back");

                    // Evaluate h_grad[hlayer_n-1][] & hidden_w[hlayer_n-1][]
                    for (int i = 0; i < hlayer[hlayer_n - 1]; i++)
                    {
                        h_grad[hlayer_n - 1][i] = 0;
                        for (int j = 0; j < category_n; j++)
                            h_grad[hlayer_n - 1][i] += o_grad[j] * out_w[j][i + 1];

                        h_grad[hlayer_n - 1][i] *= v[hlayer_n - 1][i + 1] * (1 - v[hlayer_n - 1][i + 1]);

                        //If not hidden layer 1
                        if (hlayer_n > 1)
                        {
                            for (int j = 0; j < hlayer[hlayer_n - 2] + 1; j++)
                                hidden_w[hlayer_n - 1][i][j] += learning_rate * h_grad[hlayer_n - 1][i] * v[hlayer_n - 2][j];
                        }
                        // If hidden layer 1
                        else
                        {
                            for (int j = 0; j < m - 1; j++)
                                hidden_w[0][i][j] += learning_rate * h_grad[0][i] * data[count][j];
                        }
                    }// end for

                    // Evaluate h_grad[hlayer_n-2~1][] & hidden_w[hlayer_n-2~1][]
                    for (int i = hlayer_n - 2; i > 0; i--)
                        for (int j = 0; j < hlayer[i]; j++)
                        {
                            h_grad[i][j] = 0;
                            for (int k = 0; k < hlayer[i + 1]; k++)
                                h_grad[i][j] += h_grad[i + 1][k] * hidden_w[i + 1][k][j + 1];

                            h_grad[i][j] *= v[i][j + 1] * (1 - v[i][j + 1]);

                            for (int k = 0; k < hlayer[i - 1] + 1; k++)
                                hidden_w[i][j][k] += learning_rate * h_grad[i][j] * v[i - 1][k];
                        }// end for

                    //Show2DLine(data[count][m-1]-sorted_data[0][m-1]);
                }// end for


                out_rmse = GetTrainRMSE();
                //Memo1->Lines->Add(AnsiString(it+1) + " " + AnsiString(out_rmse));
                //Series9->AddXY(it+1, out_rmse);

                // If covergence rules is RMSE
                //if (CheckBox1->Checked)
                if (out_rmse < in_rmse)
                {
                    it++;
                    break;
                }
            }// end for iteration


            // Add information to Memo1
            //Memo1->Lines->Add("");
            //Memo1->Lines->Add("Trained Weighted Vectors:");
            //PrintWeightedVectors();

            //Memo1->Lines->Add("    疊代次數 = " + AnsiString(it));
            //Memo1->Lines->Add("    訓練集 RMSE = " + AnsiString(GetTrainRMSE()));
            //Memo1->Lines->Add("    訓練集 辨識率 = " + AnsiString(GetIdentifyRate_train())+"%");

            trainrmse = GetTrainRMSE();
            traincorrect = GetIdentifyRate_train();
        }

        //---------------------------------------------------------------------------

        void Test()
        {
            if (IS_FILEOPEN == 0)
            {
                //ShowMessage("尚未開啟檔案!");
                return;
            }
            testrmse = GetTestRMSE();
            testcorrect = GetIdentifyRate();
            //Show2DPoints();
            //for (int i = 0; i < category_n; i++)
            //  Show2DLine(i);
            //ShowMissData();
            //Memo1->Lines->Add("    測試集 RMSE = " + AnsiString(GetTestRMSE()));
            //Memo1->Lines->Add("    辨識率 = " + AnsiString(GetIdentifyRate()) + "%");
        }



        //---------------------------------------------------------------------------

        double GetTrainRMSE()
        {
            int kind;
            double rmse = 0;
            double temp = 0;

            for (int i = 0; i < train_n; i++)
            {
                temp = 0;
                kind = (int)data[i][m - 1] - (int)sorted_data[0][m - 1];

                // Evaluate data[i][]
                EvaluateNoModify(i);

                for (int j = 0; j < category_n; j++)
                {
                    if (j == kind)
                        temp += Math.Sqrt((1 - out_v[j]) * (1 - out_v[j]));
                    else
                        temp += Math.Sqrt((0 - out_v[j]) * (0 - out_v[j]));
                }// end for


                rmse += temp / category_n;
            }// end for


            return rmse / train_n;
        }

        //---------------------------------------------------------------------------
        double GetTestRMSE()
        {
            int kind;
            double rmse = 0;
            double temp = 0;


            for (int i = train_n; i < n; i++)
            {
                temp = 0;
                kind = (int)data[i][m - 1] - (int)sorted_data[0][m - 1];

                // Evaluate data[i][]
                EvaluateNoModify(i);

                for (int j = 0; j < category_n; j++)
                {
                    if (j == kind)
                        temp += Math.Sqrt((1 - out_v[j]) * (1 - out_v[j]));
                    else
                        temp += Math.Sqrt((0 - out_v[j]) * (0 - out_v[j]));
                }// end for

                rmse += temp / category_n;
            }// end for


            return rmse / test_n;
        }

        //---------------------------------------------------------------------------
        double GetIdentifyRate_train()
        {
            int count = 0;
            int error = 1;
            int kind;

            for (int i = 0; i < train_n; i++)
            {
                // 設 E 不屬於任一群
                kind = -100;

                // Evaluate data[i][]
                Evaluate(i);

                for (int j = 0; j < category_n; j++)
                    if (out_v[j] >= 0.5)
                        kind = (int)sorted_data[0][m - 1] + j;


                if (kind != data[i][m - 1])
                {
                    kind = -100;

                    //for (int j = 0; j < m; j++)
                    //    Form1->StringGrid1->Cells[j + 1][error] = data[i][j];

                    //Form1->StringGrid1->Cells[m+1][error] = kind;
                    //Form1->StringGrid1->Cells[0][error] = error;
                    if (hlayer[hlayer_n - 1] == 2)
                    {
                        //Form1->StringGrid1->Cells[m + 1][error] = v[hlayer_n - 1][1];
                        //Form1->StringGrid1->Cells[m + 2][error] = v[hlayer_n - 1][2];
                    }
                    error++;
                }
                else
                    count++;
            }// end for


            return (double)count / (double)train_n * 100;
        }

        double GetIdentifyRate()
        {
            int count = 0;
            int error = 1;
            int kind;

            for (int i = train_n; i < n; i++)
            {
                // 設 E 不屬於任一群
                kind = -100;

                // Evaluate data[i][]
                Evaluate(i);

                for (int j = 0; j < category_n; j++)
                    if (out_v[j] >= 0.5)
                        kind = (int)sorted_data[0][m - 1] + j;


                if (kind != data[i][m - 1])
                {
                    kind = -100;



                    error++;
                }
                else
                    count++;
            }// end for


            return (double)count / (double)test_n * 100;
        }

        //---------------------------------------------------------------------------

        void Evaluate(int count)
        {
            // Evaluate Hidden Layer 1
            v[0][0] = -1;
            for (int i = 0; i < hlayer[0]; i++)
            {
                v[0][i + 1] = 0;
                for (int j = 0; j < m - 1; j++)
                    v[0][i + 1] += hidden_w[0][i][j] * data[count][j];

                v[0][i + 1] = 1 / (1 + Math.Exp(-v[0][i + 1]));
            }

            //  Evaluate Hidden Layer 2~hlayer_n
            for (int i = 1; i < hlayer_n; i++)
                for (int j = 0; j < hlayer[i]; j++)
                {
                    v[i][0] = -1;
                    v[i][j + 1] = 0;

                    for (int k = 0; k < hlayer[i - 1] + 1; k++)
                        v[i][j + 1] += hidden_w[i][j][k] * v[i - 1][k];

                    v[i][j + 1] = 1 / (1 + Math.Exp(-v[i][j + 1]));
                }// end for

            //  Evaluate Output Layer
            for (int i = 0; i < category_n; i++)
            {
                out_v[i] = 0;
                for (int j = 0; j < hlayer[hlayer_n - 1] + 1; j++)
                    out_v[i] += out_w[i][j] * v[hlayer_n - 1][j];

                out_v[i] = 1 / (1 + Math.Exp(-out_v[i]));
            }// end for
        }
        //---------------------------------------------------------------------------
        void EvaluateNoModify(int count)
        {
            double[][] h_v;

            // Allocate memory to **h_v
            h_v = new double[hlayer_n][];

            for (int i = 0; i < hlayer_n; i++)
                h_v[i] = new double[hlayer[i] + 1];


            // Initialize v[][]
            for (int i = 0; i < hlayer_n; i++)
                h_v[i][0] = -1;


            // Evaluate Hidden Layer 1
            for (int i = 0; i < hlayer[0]; i++)
            {
                h_v[0][i + 1] = 0;
                for (int j = 0; j < m - 1; j++)
                    h_v[0][i + 1] += hidden_w[0][i][j] * data[count][j];

                h_v[0][i + 1] = 1 / (1 + Math.Exp(-h_v[0][i + 1]));
            }

            //  Evaluate Hidden Layer 2~hlayer_n
            for (int i = 1; i < hlayer_n; i++)
                for (int j = 0; j < hlayer[i]; j++)
                {
                    h_v[i][j + 1] = 0;

                    for (int k = 0; k < hlayer[i - 1] + 1; k++)
                        h_v[i][j + 1] += hidden_w[i][j][k] * h_v[i - 1][k];

                    h_v[i][j + 1] = 1 / (1 + Math.Exp(-h_v[i][j + 1]));
                }// end for

            //  Evaluate Output Layer
            for (int i = 0; i < category_n; i++)
            {
                out_v[i] = 0;
                for (int j = 0; j < hlayer[hlayer_n - 1] + 1; j++)
                    out_v[i] += out_w[i][j] * h_v[hlayer_n - 1][j];

                out_v[i] = 1 / (1 + Math.Exp(-out_v[i]));
            }// end for

        }



        //---------------------------------------------------------------------------
    }


    
}
