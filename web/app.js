const form = document.querySelector('#avatar-form');
const idInput = document.querySelector('#user-id');
const seedInput = document.querySelector('#seed');
const tokenInput = document.querySelector('#access-token');
const counter = document.querySelector('#counter');
const submit = document.querySelector('#submit');
const statusBox = document.querySelector('#status');
const result = document.querySelector('#result');
const avatar = document.querySelector('#avatar');
const download = document.querySelector('#download');
const health = document.querySelector('#health');
const tokenField = document.querySelector('.token-field');

function tokenHeaders() {
  const token = tokenInput.value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function setStatus(message, kind = '') {
  statusBox.textContent = message;
  statusBox.className = `status ${kind}`.trim();
}

idInput.addEventListener('input', () => {
  counter.textContent = `${Array.from(idInput.value).length} / 40`;
});

async function checkHealth() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    const data = await response.json();
    health.textContent = data.busy ? '生成中' : `${data.backend.toUpperCase()} 就绪`;
    health.classList.toggle('busy', data.busy);
    tokenField.hidden = !data.token_required;
  } catch {
    health.textContent = '服务离线';
    health.classList.add('busy');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  result.hidden = true;
  submit.disabled = true;
  setStatus('正在把这个名字变成一幅画，请稍候……', 'working');
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...tokenHeaders() },
      body: JSON.stringify({ id: idInput.value, seed: Number(seedInput.value) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '生成失败');
    const imageResponse = await fetch(data.image_url, { headers: tokenHeaders(), cache: 'no-store' });
    if (!imageResponse.ok) throw new Error('无法读取生成结果');
    const blob = await imageResponse.blob();
    const objectUrl = URL.createObjectURL(blob);
    if (avatar.dataset.objectUrl) URL.revokeObjectURL(avatar.dataset.objectUrl);
    avatar.dataset.objectUrl = objectUrl;
    avatar.src = objectUrl;
    download.href = objectUrl;
    result.hidden = false;
    setStatus('');
  } catch (error) {
    setStatus(error.message || '生成失败，请稍后重试', 'error');
  } finally {
    submit.disabled = false;
    checkHealth();
  }
});

checkHealth();
setInterval(checkHealth, 15000);
