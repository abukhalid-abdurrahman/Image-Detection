using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageDetection
{
    public class ImageNetData
    {
        [LoadColumn(0)] public string ImagePath;
        [LoadColumn(1)] public string Label;

        public static IEnumerable<ImageNetData> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageNetData
                {
                    ImagePath = x[0],
                    Label = x[1]
                });
        }
    }

    public class ImageNetPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictedLables;
    }
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<ImageNetData>("images/tags.tsv", hasHeader: true);
            
        }
    }
}
