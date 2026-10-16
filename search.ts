import rawData from "./lay-k.json" with { type: "json" };
import vectorsRaw from "./lay-k-vectors.json" with { type: "json" };
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

export type Mail = {
  email: string;
  questions: string[];
  rephrased_questions: string[];
  gold_answers: string[];
};
export const data = rawData as unknown as Mail[];
const mailVectors = (vectorsRaw as unknown as { mails: number[][] }).mails;

// ── Mots-clés ──
const tokenize = (text: string): string[] =>
  text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
const mailWordSets = data.map((mail) => new Set(tokenize(mail.email)));
const docFrequency = mailWordSets
  .flatMap((words) => [...words])
  .reduce(
    (acc, word) => acc.set(word, (acc.get(word) ?? 0) + 1),
    new Map<string, number>(),
  );
const points = (word: string): number => {
  const n = docFrequency.get(word);
  return n ? Math.log(data.length / n) : 0;
};
const keywordSearch = (question: string, k: number): number[] => {
  const questionWords = tokenize(question);
  return mailWordSets
    .map((words, index) => ({
      index,
      score: questionWords
        .filter((w) => words.has(w))
        .reduce((sum, w) => sum + points(w), 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.index);
};

// ── Sens ──
const store = new MemoryVectorStore(
  new MistralAIEmbeddings({ model: "mistral-embed" }),
);
await store.addVectors(
  mailVectors,
  mailVectors.map((_, index) => ({ pageContent: "", metadata: { index } })),
);
const semanticSearch = async (vector: number[], k: number): Promise<number[]> =>
  (await store.similaritySearchVectorWithScore(vector, k)).map(
    ([doc]) => doc.metadata.index,
  );

// ── Fusion RRF ──
const fuse = (lists: number[][], k: number): number[] => {
  const scores = new Map<number, number>();
  for (const list of lists) {
    list.forEach((index, position) =>
      scores.set(index, (scores.get(index) ?? 0) + 1 / (60 + position + 1)),
    );
  }
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([index]) => index);
};

// ── Ce que les autres fichiers utilisent ──
export const hybridSearch = async (
  question: string,
  questionVector: number[],
  k = 5,
): Promise<number[]> =>
  fuse(
    [keywordSearch(question, 20), await semanticSearch(questionVector, 20)],
    k,
  );
