import { OpenAI } from 'openai';
import { RateLimitError } from 'openai/error';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

export async function handleOpenAIRequest(messages: any[]) {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await openai.chat.completions.create({
              model: "gpt-4o-mini-2024-07-18",
              messages,
              temperature: 0.7,
              stream: true,
            });

    } catch (error: any) {
      console.error(`Attempt ${attempt + 1} failed:`, error);

      if (error instanceof RateLimitError) {
        if (attempt < MAX_RETRIES - 1) {
          console.log(`Rate limit hit. Retrying in ${RETRY_DELAY}ms...`);
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          continue;
        }
        throw new Error("Rate limit exceeded. Please try again later.");
      }

      throw error;
    }
  }

  throw new Error("Max retries reached. Please try again later.");
}
