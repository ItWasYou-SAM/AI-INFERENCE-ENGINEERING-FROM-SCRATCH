// Generates ROADMAP.md from src/data/roadmap.ts (the single source of truth).
// Run: npm run roadmap
import { writeFileSync } from 'node:fs';
import { levels, allChapters } from '../src/data/roadmap.ts';
import { sources } from '../src/data/sources.ts';

const labIcon = { cpu: '🟢', 'gpu-optional': '🟡', 'gpu-required': '🔴' } as const;

let md = `# AI Inference Engineering — From Scratch: Roadmap

> Generated from \`src/data/roadmap.ts\` by \`npm run roadmap\`. Do not edit by hand.

${levels.length} levels · ${levels.reduce((n, l) => n + l.modules.length, 0)} modules · ${allChapters.length} chapters.

Each chapter follows the same arc: **Concept → Animation / interactive → Code lab → Sources → Quiz**.

Lab tags: 🟢 CPU or browser · 🟡 GPU optional · 🔴 GPU required (Colab or a rented GPU) · ✨ interactive visual

## Sources

| ID | Source | Kind |
|---|---|---|
${Object.values(sources)
  .map((s) => `| \`${s.id}\` | ${s.url ? `[${s.title}](${s.url})` : `*${s.title}*`} — ${s.author}${s.placeholder ? ' *(placeholder)*' : ''} | ${s.kind} |`)
  .join('\n')}
`;

for (const level of levels) {
  md += `\n## Level ${level.num} — ${level.name}: "${level.question}"\n`;
  for (const m of level.modules) {
    const srcs = m.sources.map((s) => `\`${s.id}\`${s.note ? ` (${s.note})` : ''}`).join(', ');
    md += `\n### ${m.id === 'capstone' ? 'Capstone' : `M${m.num}`} · ${m.title}\n\n_${m.tagline}_\n\n**Sources:** ${srcs}\n\n`;
    for (const ch of m.chapters) {
      const tags = [ch.lab ? labIcon[ch.lab] : '', ch.interactive ? '✨' : ''].filter(Boolean).join(' ');
      md += `- **${ch.id}** ${ch.title}${tags ? ` ${tags}` : ''}\n`;
    }
  }
}

writeFileSync(new URL('../ROADMAP.md', import.meta.url), md);
console.log(`ROADMAP.md written: ${allChapters.length} chapters`);
