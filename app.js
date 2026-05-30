// ============================================================
// 照片去重工具 v3.0 — 类别感知前端逻辑
// ============================================================

let selectedPath = '';
let scanResult = null;
let selectedToDelete = new Set();
let currentBrowserPath = '/';
let progressTimer = null;

function updateThreshold() {
  const val = document.getElementById('threshold').value;
  document.getElementById('thresholdValue').textContent = val + '%';
  document.querySelectorAll('.tag').forEach(t => t.classList.remove('tag-active'));
  const active = document.querySelector(`.tag[data-threshold="${val}"]`);
  if (active) active.classList.add('tag-active');
}

function setProgress(pct, text) {
  document.getElementById('progressBar').style.display = 'block';
  document.getElementById('progressFill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
  document.getElementById('progressText').textContent = text;
}

async function startScan() {
  const folder = document.getElementById('folderPath').value.trim();
  if (!folder) { alert('请输入文件夹路径'); return; }
  const threshold = parseInt(document.getElementById('threshold').value) / 100;

  scanResult = null;
  selectedToDelete.clear();
  document.getElementById('results').style.display = 'none';
  document.getElementById('emptyState').style.display = 'none';
  setProgress(3, '准备扫描...');

  try {
    const resp = await fetch('/scan_start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder, threshold })
    });
    const start = await resp.json();
    if (start.error) throw new Error(start.error);
    pollScan(start.job_id);
  } catch (err) {
    document.getElementById('progressBar').style.display = 'none';
    showError(`扫描启动失败: ${err.message}`);
  }
}

async function pollScan(jobId) {
  if (progressTimer) clearTimeout(progressTimer);
  try {
    const resp = await fetch(`/scan_status?job_id=${encodeURIComponent(jobId)}&_=${Date.now()}`);
    const job = await resp.json();
    setProgress(job.progress || 0, job.message || '扫描中...');

    if (job.status === 'done') {
      scanResult = job.result;
      selectedToDelete.clear();
      setProgress(100, '扫描完成！');
      setTimeout(() => {
        document.getElementById('progressBar').style.display = 'none';
        renderResults(job.result);
      }, 350);
      return;
    }
    if (job.status === 'error' || job.status === 'missing') {
      document.getElementById('progressBar').style.display = 'none';
      showError(job.error || '扫描任务异常');
      return;
    }
    progressTimer = setTimeout(() => pollScan(jobId), 650);
  } catch (err) {
    document.getElementById('progressBar').style.display = 'none';
    showError(`读取进度失败: ${err.message}`);
  }
}

function showError(msg) {
  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'block';
  resultsEl.innerHTML = `<div class="error-msg">❌ ${escapeHtml(msg)}</div>`;
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderResults(data) {
  const resultsEl = document.getElementById('results');
  if (!data) return showError('没有扫描结果');
  if (data.error) return showError(data.error);

  const groups = data.groups || [];
  const totalGroups = groups.length;
  const totalFiles = groups.reduce((sum, g) => sum + g.files.length, 0);
  const categoryHtml = renderCategorySummary(data.categories || {});

  resultsEl.innerHTML = `
    <div class="summary">
      <div class="summary-item"><div class="summary-number">${data.total_images || 0}</div><div class="summary-label">总照片数</div></div>
      <div class="summary-item"><div class="summary-number">${totalGroups}</div><div class="summary-label">重复/相似组</div></div>
      <div class="summary-item"><div class="summary-number">${totalFiles}</div><div class="summary-label">涉及文件</div></div>
      <div class="summary-item"><div class="summary-number">${Math.round(data.threshold || 70)}%</div><div class="summary-label">相似度阈值</div></div>
    </div>
    ${categoryHtml}
    <div class="group-actions">
      <button class="btn btn-secondary" onclick="selectAll()">☑ 全选删除项</button>
      <button class="btn btn-secondary" onclick="deselectAll()">☐ 取消全选</button>
      <button class="btn btn-danger" id="deleteBtn" onclick="deleteSelected()" disabled>🗑 删除选中 (<span id="deleteCount">0</span>)</button>
    </div>
    <div class="groups" id="groups"></div>
  `;
  resultsEl.style.display = 'block';

  const groupsEl = document.getElementById('groups');
  if (groups.length === 0) {
    groupsEl.innerHTML = `<div class="empty-state"><div class="empty-icon">✅</div><p>未发现重复或相似的照片！类别识别已完成。</p></div>`;
    return;
  }

  groups.forEach((group, gIdx) => renderGroup(groupsEl, group, gIdx));
}

function renderCategorySummary(categories) {
  const entries = Object.entries(categories);
  if (entries.length === 0) return '';
  const chips = entries.map(([name, count]) => `<span class="category-chip">${escapeHtml(name)} <b>${count}</b></span>`).join('');
  return `<div class="category-summary"><div class="category-title">类别分布</div><div>${chips}</div></div>`;
}

function detailBars(details) {
  const labels = {hash:'指纹', color:'颜色', structure:'结构', edge:'轮廓', layout:'布局', category:'类别'};
  return Object.entries(labels).map(([key, label]) => {
    const val = Math.round((details && details[key]) || 0);
    return `<div class="detail-bar"><span>${label}</span><div><i style="width:${val}%"></i></div><em>${val}%</em></div>`;
  }).join('');
}

function renderGroup(groupsEl, group, gIdx) {
  const isExact = group.type === '相同文件';
  const badgeClass = isExact ? 'badge-exact' : 'badge-similar';
  const simClass = group.similarity >= 90 ? 'sim-high' : group.similarity >= 70 ? 'sim-mid' : 'sim-low';

  let cardsHtml = '';
  group.files.forEach((file, fIdx) => {
    const isKeep = file.path === group.keep || fIdx === 0;
    const cardClass = isKeep ? 'keep' : 'to-delete';
    const previewUrl = `/preview/${encodeURI(file.path)}`;
    const safePath = escapeHtml(file.path).replace(/'/g, '&#39;');
    cardsHtml += `
      <div class="image-card ${cardClass}" id="card-${gIdx}-${fIdx}">
        <div class="image-checkbox" onclick="toggleFile(${gIdx}, ${fIdx}, '${safePath}')">✓</div>
        <span class="image-badge ${isKeep ? 'badge-keep' : 'badge-delete'}">${isKeep ? '✓ 保留' : '🗑 待删'}</span>
        <img src="${previewUrl}" alt="${escapeHtml(file.name)}" loading="lazy" onerror="this.style.display='none'">
        <div class="card-footer">
          <span class="filename">${escapeHtml(file.name)}</span>
          <span class="file-meta">${escapeHtml(file.size)} · ${file.file_size_kb}KB</span>
          <span class="file-category">${escapeHtml(file.category_icon || '📷')} ${escapeHtml(file.category || '未知')} · ${file.category_confidence || 0}%</span>
        </div>
      </div>`;
  });

  const card = document.createElement('div');
  card.className = 'group-card';
  card.innerHTML = `
    <div class="group-header">
      <div class="group-header-left">
        <span class="group-type-badge ${badgeClass}">${group.type}</span>
        <span class="group-category">${escapeHtml(group.category_icon || '📷')} ${escapeHtml(group.category || '未知')}</span>
        <span class="group-similarity ${simClass}">${group.similarity}%</span>
      </div>
      <span class="group-count">${group.files.length} 张</span>
    </div>
    <div class="similarity-details">${detailBars(group.details || {})}</div>
    <div class="image-grid">${cardsHtml}</div>`;
  groupsEl.appendChild(card);
}

function toggleFile(gIdx, fIdx, path) {
  const card = document.getElementById(`card-${gIdx}-${fIdx}`);
  if (!scanResult || !scanResult.groups[gIdx]) return;
  const group = scanResult.groups[gIdx];
  const file = group.files[fIdx];
  if (!file || file.path === group.keep || fIdx === 0) return;
  path = file.path;
  if (selectedToDelete.has(path)) {
    selectedToDelete.delete(path);
    if (card) card.classList.remove('selected');
  } else {
    selectedToDelete.add(path);
    if (card) card.classList.add('selected');
  }
  updateDeleteBtn();
}

function selectAll() {
  if (!scanResult) return;
  scanResult.groups.forEach((group, gIdx) => {
    group.files.forEach((file, fIdx) => {
      if (file.path !== group.keep && fIdx > 0) {
        selectedToDelete.add(file.path);
        const card = document.getElementById(`card-${gIdx}-${fIdx}`);
        if (card) card.classList.add('selected');
      }
    });
  });
  updateDeleteBtn();
}

function deselectAll() {
  selectedToDelete.clear();
  document.querySelectorAll('.image-card.selected').forEach(el => el.classList.remove('selected'));
  updateDeleteBtn();
}

function updateDeleteBtn() {
  const btn = document.getElementById('deleteBtn');
  const count = document.getElementById('deleteCount');
  if (btn && count) {
    count.textContent = selectedToDelete.size;
    btn.disabled = selectedToDelete.size === 0;
  }
}

async function deleteSelected() {
  if (selectedToDelete.size === 0) return;
  if (!confirm(`确定要删除 ${selectedToDelete.size} 张照片吗？（将移到废纸篓）`)) return;
  try {
    const resp = await fetch('/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: Array.from(selectedToDelete) })
    });
    const data = await resp.json();
    if (data.deleted > 0) {
      alert(`✅ 已删除 ${data.deleted} 张照片（已移至废纸篓）`);
      startScan();
    }
    if (data.errors && data.errors.length > 0) alert(`⚠️ ${data.errors.length} 张删除失败`);
  } catch (err) {
    alert(`❌ 删除失败: ${err.message}`);
  }
}

function triggerUpload() { document.getElementById('fileInput').click(); }

async function handleUpload(files) {
  if (files.length === 0) return;
  const formData = new FormData();
  for (const file of files) formData.append('files', file);
  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    alert(`✅ 已上传 ${data.uploaded} 张照片`);
    document.getElementById('folderPath').value = data.folder || 'photos';
    startScan();
  } catch (err) {
    alert(`❌ 上传失败: ${err.message}`);
  }
}

function openBrowser() {
  document.getElementById('browserModal').style.display = 'flex';
  currentBrowserPath = '/Users/jieliuzi';
  loadFolders(currentBrowserPath);
}
function closeBrowser() { document.getElementById('browserModal').style.display = 'none'; }

async function loadFolders(path) {
  const listEl = document.getElementById('folderList');
  listEl.innerHTML = '<div class="loading" style="padding:20px;text-align:center;color:#9ca3af">加载中...</div>';
  try {
    const resp = await fetch(`/listdir?path=${encodeURIComponent(path)}`);
    const data = await resp.json();
    if (data.error) { listEl.innerHTML = `<div class="error-msg">${escapeHtml(data.error)}</div>`; return; }
    currentBrowserPath = data.current;
    document.getElementById('currentPath').textContent = data.current;
    document.getElementById('selectBtn').disabled = false;
    let html = '';
    if (data.parent) {
      html += `<div class="folder-item" ondblclick="loadFolders('${data.parent.replace(/'/g, "\\'")}')"><span class="folder-item-icon">📁</span><span class="folder-item-name">.. 返回上级</span></div>`;
    }
    data.items.forEach(item => {
      if (item.is_dir) {
        html += `<div class="folder-item" onclick="highlightFolder(this)" ondblclick="loadFolders('${item.path.replace(/'/g, "\\'")}')" data-path="${escapeHtml(item.path)}"><span class="folder-item-icon">📁</span><span class="folder-item-name">${escapeHtml(item.name)}</span></div>`;
      }
    });
    const imageCount = data.items.filter(i => !i.is_dir).length;
    if (imageCount > 0) {
      html += `<div class="folder-item" style="border-top:1px solid #e5e7eb;margin-top:4px;padding-top:10px" onclick="highlightFolder(this)" data-path="${escapeHtml(data.current)}"><span class="folder-item-icon">🖼</span><span class="folder-item-name" style="color:#10b981">包含 ${imageCount} 张图片</span></div>`;
    }
    listEl.innerHTML = html || '<div style="padding:20px;text-align:center;color:#9ca3af">空文件夹</div>';
  } catch (err) {
    listEl.innerHTML = `<div class="error-msg">加载失败: ${escapeHtml(err.message)}</div>`;
  }
}

function highlightFolder(el) {
  document.querySelectorAll('.folder-item.selected').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
  document.getElementById('selectBtn').disabled = false;
}
function selectFolder() {
  const selected = document.querySelector('.folder-item.selected');
  if (selected) {
    const path = selected.getAttribute('data-path');
    document.getElementById('folderPath').value = path;
    selectedPath = path;
  } else {
    document.getElementById('folderPath').value = currentBrowserPath;
    selectedPath = currentBrowserPath;
  }
  closeBrowser();
}
function clearPath() {
  document.getElementById('folderPath').value = '';
  selectedPath = '';
}

document.addEventListener('DOMContentLoaded', updateThreshold);
