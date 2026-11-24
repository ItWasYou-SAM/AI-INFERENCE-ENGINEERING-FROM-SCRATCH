import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

/**
 * Chapter bodies. The file path decides the id: src/content/chapters/m09/kv-cache-size.mdx → "m09/kv-cache-size",
 * which must match a chapter in src/data/roadmap.ts. Titles and ordering live in roadmap.ts.
 */
const chapters = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/chapters' }),
  schema: z.object({
    summary: z.string(),
    duration: z.string().optional(), // e.g. "25 min"
    objectives: z.array(z.string()).default([]),
  }),
});

export const collections = { chapters };
