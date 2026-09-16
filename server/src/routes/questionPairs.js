import { Router } from "express";
import { randomUUID } from "node:crypto";
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
      _id: randomUUID(),
      question1,
      question2,
      ...analysis,
      createdAt: new Date().toISOString(),
    };

    memoryHistory.unshift(payload);
    if (memoryHistory.length > 20) {
      memoryHistory.splice(20);
    }

    return res.status(201).json(payload);
  } catch (error) {
    return next(error);
  }
});

router.get("/history", async (_req, res, next) => {
  try {
    return res.json(memoryHistory.slice(0, 8));
  } catch (error) {
    return next(error);
  }
});

export default router;
