import rawData from "./lay-k.json" with { type: "json" };
import { writeFileSync } from "node:fs";
import { MistralAIEmbeddings } from "@langchain/mistralai";

type Mail = {
  email: string;
  questions: string[];
  rephrased_questions: string[];
};
const data = rawData as unknown as Mail[];

const embeddings = new MistralAIEmbeddings({ model: "mistral-embed" });
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Un paquet, avec nouvel essai si Mistral répond 429
const embedWithRetry = async (texts: string[]): Promise<number[][]> => {
  for (let attempt = 1; ; attempt++) {
    try {
      return await embeddings.embedDocuments(texts);
    } catch (error: any) {
      if (error.statusCode !== 429 || attempt === 5) throw error;
      console.log(`Limite atteinte, pause de ${attempt * 5} s…`);
      await sleep(attempt * 5000);
    }
  }
};

// Tous les textes, par paquets, avec une pause entre chaque paquet
const embedAll = async (texts: string[], batchSize: number, label: string) => {
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    vectors.push(...(await embedWithRetry(texts.slice(i, i + batchSize))));
    console.log(
      `${label} : ${Math.min(i + batchSize, texts.length)}/${texts.length}`,
    );
    await sleep(1500);
  }
  return vectors;
};

// Le même découpage que dans explorer.ts, en gardant une question sur 11
const sample = (field: "questions" | "rephrased_questions") =>
  data
    .flatMap((mail, index) =>
      mail[field].map((question) => ({ question, expected: index })),
    )
    .filter((_, i) => i % 11 === 0);

const questions = sample("questions");
const rephrased = sample("rephrased_questions");

// Les mails sont longs : petits paquets. Les questions sont courtes : gros paquets.
const mailVectors = await embedAll(
  data.map((m) => m.email),
  16,
  "Mails",
);
const questionVectors = await embedAll(
  questions.map((q) => q.question),
  64,
  "Questions",
);
const rephrasedVectors = await embedAll(
  rephrased.map((q) => q.question),
  64,
  "Reformulées",
);

writeFileSync(
  "lay-k-vectors.json",
  JSON.stringify({
    mails: mailVectors,
    questions: questions.map((q, i) => ({ ...q, vector: questionVectors[i] })),
    rephrased: rephrased.map((q, i) => ({ ...q, vector: rephrasedVectors[i] })),
  }),
);
console.log("Sauvegardé dans lay-k-vectors.json");
