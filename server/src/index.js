import dotenv from "dotenv";
import app from "./server.js";

dotenv.config();

const port = process.env.PORT || 5001;

app.listen(port, "127.0.0.1", () => {
  console.log(`API running on http://127.0.0.1:${port}`);
});
