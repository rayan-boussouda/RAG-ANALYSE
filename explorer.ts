import rawData from "./lay-k.json" with { type: "json" };

type Mail = {
  path: string;
  email: string;
  questions: string[];
  rephrased_questions: string[];
  gold_answers: string[];
};

type TestCase = {
  question: string;
  expected: number;
};

type SearchResult = {
  index: number;
  score: number;
};

const data = rawData as unknown as Mail[];

const tokenize = (text: string): string[] =>
  text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

const mailWordSets: Set<string>[] = data.map(
  (mail) => new Set(tokenize(mail.email)),
);

const buildDocFrequency = (
  mailWordSets: Set<string>[],
): Map<string, number> => {
  const allWords: string[] = mailWordSets.flatMap((mailWords) => [
    ...mailWords,
  ]);

  const occurrences = allWords.reduce(
    (acc: Map<string, number>, word: string) => {
      if (!acc.has(word)) {
        acc.set(word, 0);
      }
      acc.set(word, acc.get(word)! + 1);
      return acc;
    },
    new Map<string, number>(),
  );

  return occurrences;
};

const points = (
  word: string,
  docFrequency: Map<string, number>,
  totalMails: number,
): number => {
  const mailsWithWord = docFrequency.get(word);
  return mailsWithWord ? Math.log(totalMails / mailsWithWord) : 0;
};

const docFrequency = buildDocFrequency(mailWordSets);
const totalMails = data.length;

const search = (question: string, k = 5): SearchResult[] => {
  const questionWords = tokenize(question);
  return mailWordSets
    .map((words, index) => ({
      index,
      score: questionWords
        .filter((w) => words.has(w))
        .reduce((sum, w) => sum + points(w, docFrequency, totalMails), 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
};

const buildTestSet = (field: "questions" | "rephrased_questions"): TestCase[] =>
  data
    .flatMap((mail, index) =>
      mail[field].map((question) => ({ question, expected: index })),
    )
    .filter((_, i) => i % 11 === 0);

const recallAt5 = (testSet: TestCase[]): string => {
  const found = testSet.filter(({ question, expected }) =>
    search(question).some((r) => r.index === expected),
  ).length;
  return `${((found / testSet.length) * 100).toFixed(1)} % (${found}/${testSet.length})`;
};

console.log("Questions   :", recallAt5(buildTestSet("questions")));
console.log("Reformulées :", recallAt5(buildTestSet("rephrased_questions")));
