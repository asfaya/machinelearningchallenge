using Microsoft.ML.Data;

namespace BairesDevCodingChallenge
{
    public class LinkedInProfiles
    {
        [Column(ordinal: "0", name: "Label")]
        public float SendMail;
        [Column(ordinal: "1")]
        public string PersonId { get; set; }
        [Column(ordinal: "4")]
        public string CurrentRole { get; set; }
        [Column(ordinal: "5")]
        public string Country { get; set; }
        [Column(ordinal: "6")]
        public string Industry { get; set; }
    }

    public class LinkedInPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
