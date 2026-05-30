import { Router } from "express";
import { randomUUID } from "node:crypto";
import { isDatabaseConnected } from "../lib/database.js";
import QuestionPair from "../models/QuestionPair.js";
import { predictWithFlaskModel } from "../services/flaskModel.js";

const router = Router();
const memoryHistory = [];

router.post("/predict", async (req, res, next) => {
  try {
    const question1 = String(req.body.question1 || "").trim();
    const question2 = String(req.body.question2 || "").trim();

    if (!question1 || !question2) {
      return res.status(400).json({ message: "Both questions are required." });
    }

    const analysis = await predictWithFlaskModel(question1, question2);
    const payload = {
      question1,
      question2,
      ...analysis,
    };

    if (isDatabaseConnected()) {
      const savedPair = await QuestionPair.create(payload);
      return res.status(201).json(savedPair);
    }

    const fallbackPair = {
      _id: randomUUID(),
      ...payload,
      createdAt: new Date().toISOString(),
    };
    memoryHistory.unshift(fallbackPair);
    memoryHistory.splice(8);

    return res.status(201).json(fallbackPair);
  } catch (error) {
    return next(error);
  }
});

router.get("/history", async (_req, res, next) => {
  try {
    if (isDatabaseConnected()) {
      const pairs = await QuestionPair.find().sort({ createdAt: -1 }).limit(8);
      return res.json(pairs);
    }

    return res.json(memoryHistory);
  } catch (error) {
    return next(error);
  }
});

export default router;
