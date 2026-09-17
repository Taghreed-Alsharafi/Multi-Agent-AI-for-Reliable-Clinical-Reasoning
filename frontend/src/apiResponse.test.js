import assert from 'node:assert/strict';
import test from 'node:test';
import { readApiResponse } from './apiResponse.js';

test('hosting errors never surface JSON parser messages', async () => {
  await assert.rejects(
    readApiResponse(new Response('A server error has occurred', { status: 500 })),
    /temporarily unavailable/,
  );
});

test('hosting timeout is explained', async () => {
  await assert.rejects(readApiResponse(new Response('timeout', { status: 504 })), /time limit/);
});

test('structured backend errors remain readable', async () => {
  await assert.rejects(
    readApiResponse(new Response(JSON.stringify({ detail: 'Service is not configured' }), { status: 503 })),
    /not configured/,
  );
});

test('successful JSON is returned', async () => {
  assert.deepEqual(await readApiResponse(new Response('{"status":"ok"}')), { status: 'ok' });
});
