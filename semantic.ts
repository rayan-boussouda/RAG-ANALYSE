import vectorsRaw from "./lay-k-vectors.json" with { type: "json" };
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

type TestCase = { question: string; expected: number; vector: number[] };
type Vectors = {
  mails: number[][];
  questions: TestCase[];
  rephrased: TestCase[];
};
const vectors = vectorsRaw as unknown as Vectors;

// On recharge les vecteurs déjà calculés : aucun appel à Mistral ici
const store = new MemoryVectorStore(
  new MistralAIEmbeddings({ model: "mistral-embed" }),
);
await store.addVectors(
  vectors.mails,
  vectors.mails.map((_, index) => ({ pageContent: "", metadata: { index } })),
);

const recallAt5 = async (testSet: TestCase[]) => {
  let found = 0;
  for (const { vector, expected } of testSet) {
    // On cherche directement avec le vecteur déjà calculé de la question
    const results = await store.similaritySearchVectorWithScore(vector, 5);
    if (results.some(([doc]) => doc.metadata.index === expected)) found++;
  }
  return `${((found / testSet.length) * 100).toFixed(1)} % (${found}/${testSet.length})`;
};

console.log("Questions   :", await recallAt5(vectors.questions));
console.log("Reformulées :", await recallAt5(vectors.rephrased));
