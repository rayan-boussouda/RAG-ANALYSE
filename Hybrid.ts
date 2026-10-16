import rawData from "./lay-k.json" with { type: "json" };
import vectorsRaw from "./lay-k-vectors.json" with { type: "json" };
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

type Mail = { email: string };
type TestCase = { question: string; expected: number; vector: number[] };
type Vectors = {
  mails: number[][];
  questions: TestCase[];
  rephrased: TestCase[];
};

const data = rawData as unknown as Mail[];
const vectors = vectorsRaw as unknown as Vectors;

// ─────────── 1. Recherche par mots-clés (ton code d'explorer.ts) ───────────
const tokenize = (text: string): string[] =>
  text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

const mailWordSets = data.map((mail) => new Set(tokenize(mail.email)));

const docFrequency = mailWordSets
  .flatMap((words) => [...words])
  .reduce(
    (acc, word) => acc.set(word, (acc.get(word) ?? 0) + 1),
    new Map<string, number>(),
  );

const totalMails = data.length;
const points = (word: string): number => {
  const mailsWithWord = docFrequency.get(word);
  return mailsWithWord ? Math.log(totalMails / mailsWithWord) : 0;
};

// Renvoie les index des k meilleurs mails
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

// ─────────── 2. Recherche par sens (vecteurs déjà calculés) ───────────
const store = new MemoryVectorStore(
  new MistralAIEmbeddings({ model: "mistral-embed" }),
);
await store.addVectors(
  vectors.mails,
  vectors.mails.map((_, index) => ({ pageContent: "", metadata: { index } })),
);

const semanticSearch = async (vector: number[], k: number): Promise<number[]> =>
  (await store.similaritySearchVectorWithScore(vector, k)).map(
    ([doc]) => doc.metadata.index,
  );

// ─────────── 3. Fusion RRF ───────────
// Chaque mail gagne 1 / (60 + position) dans chaque liste où il apparaît
const fuse = (lists: number[][], k = 5): number[] => {
  const scores = new Map<number, number>();
  for (const list of lists) {
    list.forEach((index, position) => {
      scores.set(index, (scores.get(index) ?? 0) + 1 / (60 + position + 1));
    });
  }
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([index]) => index);
};

// ─────────── 4. Évaluation des 3 méthodes ───────────
const evaluate = async (testSet: TestCase[], label: string) => {
  let keyword = 0,
    semantic = 0,
    hybrid = 0,
    onlyKeyword = 0,
    onlySemantic = 0;

  for (const { question, vector, expected } of testSet) {
    const keywordList = keywordSearch(question, 20);
    const semanticList = await semanticSearch(vector, 20);

    const keywordOk = keywordList.slice(0, 5).includes(expected);
    const semanticOk = semanticList.slice(0, 5).includes(expected);
    const hybridOk = fuse([keywordList, semanticList]).includes(expected);

    if (keywordOk) keyword++;
    if (semanticOk) semantic++;
    if (hybridOk) hybrid++;
    if (keywordOk && !semanticOk) onlyKeyword++;
    if (semanticOk && !keywordOk) onlySemantic++;
  }

  const pct = (n: number) => `${((n / testSet.length) * 100).toFixed(1)} %`;
  console.log(`\n${label}`);
  console.log("  Mots-clés :", pct(keyword));
  console.log("  Sens      :", pct(semantic));
  console.log("  Hybride   :", pct(hybrid));
  console.log("  Trouvées seulement par les mots-clés :", onlyKeyword);
  console.log("  Trouvées seulement par le sens       :", onlySemantic);
};

await evaluate(vectors.questions, "Questions d'origine");
await evaluate(vectors.rephrased, "Reformulées");
