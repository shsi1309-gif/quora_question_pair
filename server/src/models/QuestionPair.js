import mongoose from "mongoose";

const questionPairSchema = new mongoose.Schema(
  {
    question1: {
      type: String,
      required: true,
      trim: true,
    },
    question2: {
      type: String,
      required: true,
      trim: true,
    },
    prediction: {
      type: String,
      enum: ["duplicate", "not_duplicate"],
      required: true,
    },
    confidence: {
      type: Number,
      min: 0,
      max: 100,
    },
    features: {
      type: Object,
    },
    modelSource: {
      type: String,
      default: "python_pickle_model",
    },
  },
  { timestamps: true },
);

export default mongoose.model("QuestionPair", questionPairSchema);
