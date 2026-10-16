import { createInterface } from "node:readline/promises";
import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { data, hybridSearch } from "./search.ts";

const embeddings = new MistralAIEmbeddings({ model: "mistral-embed" });
const llm = new ChatMistralAI({ model: "ministral-8b-latest", temperature: 0 });
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Exécute un appel ; si Mistral répond 429, attend puis réessaie (jusqu'à 5 fois)
const withRetry = async <T>(call: () => Promise<T>): Promise<T> => {
  for (let attempt = 1; ; attempt++) {
    try {
      return await call();
    } catch (error: any) {
      if (error.statusCode !== 429 || attempt === 5) throw error;
      console.log(`Limite Mistral atteinte, pause de ${attempt * 3} s…`);
      await sleep(attempt * 3000);
    }
  }
};

const ask = async (question: string) => {
  // ── R : trouver les 5 mails les plus pertinents ──
  const questionVector = await withRetry(() => embeddings.embedQuery(question));
  const indexes = await hybridSearch(question, questionVector);

  // ── Construire le prompt : consigne + 5 mails + question ──
  const context = indexes
    .map(
      (index, n) =>
        `--- Email ${n + 1} (id ${index}) ---\n${data[index].email}`,
    )
    .join("\n\n");

  const prompt = `You answer questions about Kenneth Lay's mailbox.
Use ONLY the emails below. If the answer is not in them, say "I cannot find this information".
Cite the id of the email you used.

${context}

Question: ${question}`;

  // ── G : le LLM lit les 5 mails et répond ──
  const response = await withRetry(() => llm.invoke(prompt));

  return {
    indexes,
    answer: response.content,
    promptTokens: Math.round(prompt.length / 4),
  };
};

// ── Interface dans le terminal : tu poses tes questions ──
const rl = createInterface({ input: process.stdin, output: process.stdout });
console.log("Exemple de question :", data[0].questions[0]);

while (true) {
  const question = await rl.question('\nTa question (ou "exit") : ');
  if (question === "exit") break;

  const result = await ask(question);
  console.log("Mails utilisés :", result.indexes);
  console.log(
    "Tokens envoyés :",
    result.promptTokens,
    "(contre ~634 000 sans RAG)",
  );
  console.log("Réponse        :", result.answer);
}
rl.close();
