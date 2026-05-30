import dotenv from "dotenv";
import app from "./server.js";
import { connectDatabase } from "./lib/database.js";

dotenv.config();

const port = process.env.PORT || 5001;

await connectDatabase(process.env.MONGODB_URI);

app.listen(port, "127.0.0.1", () => {
  console.log(`API running on http://127.0.0.1:${port}`);
});
