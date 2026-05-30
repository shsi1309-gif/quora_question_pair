import mongoose from "mongoose";

export async function connectDatabase(uri) {
  if (!uri) {
    console.warn("MONGODB_URI is not set. API will run without persistence.");
    return;
  }

  try {
    await mongoose.connect(uri);
    console.log("MongoDB connected");
  } catch (error) {
    console.warn(`MongoDB connection failed: ${error.message}`);
    console.warn("API will continue without saving comparison history.");
  }
}

export function isDatabaseConnected() {
  return mongoose.connection.readyState === 1;
}
