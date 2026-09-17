export async function readApiResponse(response) {
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    const message = response.status === 504
      ? 'The assessment exceeded the hosting time limit. Please try a shorter case.'
      : 'The assessment service is temporarily unavailable. Please try again shortly.';
    throw new Error(message);
  }
  if (!response.ok) {
    throw new Error(
      typeof payload?.detail === 'string'
        ? payload.detail
        : 'The assessment could not be completed. Please try again.',
    );
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('The assessment service returned an invalid response. Please try again.');
  }
  return payload;
}
