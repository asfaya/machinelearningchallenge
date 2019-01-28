using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace BairesDevCodingChallenge
{
    class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "people-train.in");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "people-test.in");
        private static readonly string _inputDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "people.in");
        private static readonly string _outputDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "people.out");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        private static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = "|",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("CurrentRole", DataKind.Text, 4),
                    new TextLoader.Column("Country", DataKind.Text, 5),
                    new TextLoader.Column("Industry", DataKind.Text, 6),
                }
            });

            // var trainingPipeline = BuildAndTrainModel(mlContext, _trainingDataView, pipeline);

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            PredictWithModelLoadedFromFile(mlContext);

            Console.ReadLine();
        }

        private static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("CurrentRole", "CurrentRoleFeaturized")
                                .Append(mlContext.Transforms.Text.FeaturizeText("Country", "CountryFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText("Industry", "IndustryFeaturized"))
                                .Append(mlContext.Transforms.Concatenate("Features", "CurrentRoleFeaturized", "IndustryFeaturized"))
                                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = model.Transform(dataView);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            SaveModelAsFile(mlContext, model);
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        private static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            var profiles = GetFromFile();

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine
            var profilesStreamingDataView = mlContext.CreateStreamingDataView(profiles);
            var predictions = loadedModel.Transform(profilesStreamingDataView);

            // Use the model to predict whether should or not send the email
            var predictedResults = predictions.AsEnumerable<LinkedInPrediction>(mlContext, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

            var profilesAndPredictions = profiles.Zip(predictedResults, (profile, prediction) => (profile, prediction));

            var results = (from p in profilesAndPredictions
                           where
                            p.prediction.Prediction
                           orderby p.prediction.Probability descending
                           select p).Take(100);

            SaveFile(results);
            foreach (var item in results)
            {
                Console.WriteLine($"Profile: {item.profile.PersonId} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Send" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");

        }

        private static IEnumerable<LinkedInProfiles> GetFromFile()
        {
            var list = new List<LinkedInProfiles>();
            try
            {
                //Pass the file path and file name to the StreamReader constructor
                StreamReader sr = new StreamReader(_inputDataPath);

                //Read the first line of text
                var line = sr.ReadLine();

                //Continue to read until you reach end of file
                while (!string.IsNullOrEmpty(line))
                {
                    var fields = line.Split('|');
                    list.Add(new LinkedInProfiles()
                    {
                        PersonId = fields[0],
                        CurrentRole = fields[3],
                        Country = fields[4],
                        Industry = fields[5]
                    });

                    line = sr.ReadLine();
                }

                //close the file
                sr.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }

            return list;
        }

        private static void SaveFile(IEnumerable<(LinkedInProfiles, LinkedInPrediction)> results)
        {
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(_outputDataPath))
            {
                foreach (var result in results)
                {
                    file.WriteLine(result.Item1.PersonId);
                }
            }
        }
    }
}
