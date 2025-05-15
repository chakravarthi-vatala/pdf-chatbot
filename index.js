import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { AzureChatOpenAI, AzureOpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

app.post("/ask", upload.single("pdf"), async (req, res) => {
  try {
    const query = req.body.query;
    const pdfPath = req.file.path;

    console.log("Loading PDF...");
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();

    console.log("Splitting PDF...");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const output = await splitter.splitDocuments(docs);

    const embeddings = new AzureOpenAIEmbeddings({
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE,
      azureOpenAIApiEmbeddingsDeploymentName: process.env.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
      azureOpenAIApiVersion: process.env.AZURE_OPENAI_VERSION,
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(
      output,
      embeddings
    );
    const retriever = vectorStore.asRetriever();
    const model = new AzureChatOpenAI({
      temperature: 0.5,
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE,
      azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_DEPLOYMENT,
      azureOpenAIApiVersion: process.env.AZURE_OPENAI_VERSION,
    });

    const chain = RetrievalQAChain.fromLLM(model, retriever);
    const answer = await chain.call({ query });

    fs.unlinkSync(pdfPath);

    res.json({ answer: answer.text });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Failed to process PDF and query." });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
