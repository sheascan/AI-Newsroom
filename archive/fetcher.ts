export async function fetchJson<T = any>(
  url: string,
  options?: RequestInit & { timeoutMs?: number }
): Promise<T> {
  const timeoutMs = options && (options as any).timeoutMs;
  const { timeoutMs: _ignored, ...fetchOptions } = (options || {}) as any;
  const controller = new AbortController();
  const signal = controller.signal;
  if (!fetchOptions.signal) fetchOptions.signal = signal;

  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  if (typeof timeoutMs === 'number' && timeoutMs > 0) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }

  try {
    const res = await fetch(url, fetchOptions as RequestInit);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return (await res.json()) as T;
    return (await res.text()) as unknown as T;
  } catch (err: any) {
    if (err && (err.name === 'AbortError' || err.type === 'aborted')) {
      throw new Error(`Request aborted${timeoutMs ? ` after ${timeoutMs}ms` : ''}`);
    }
    throw err;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export default fetchJson;
