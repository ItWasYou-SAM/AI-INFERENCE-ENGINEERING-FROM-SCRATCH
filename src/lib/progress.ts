const KEY = 'inference-progress-v1';

export function getDone(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(KEY) ?? '[]') as string[]);
  } catch {
    return new Set();
  }
}

export function setDone(chapterKey: string, done: boolean) {
  const s = getDone();
  if (done) s.add(chapterKey);
  else s.delete(chapterKey);
  localStorage.setItem(KEY, JSON.stringify([...s]));
  window.dispatchEvent(new CustomEvent('progresschange'));
}
