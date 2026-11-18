/**
 * FairHire AI 3.0 — Frontend Interactivity & Client-Side Engine
 * Handles tab navigation, dynamic candidate management, async resume parsing,
 * live weight sliders, and interactive pairwise XAI comparisons.
 */

// Safe HTML escaping helper
function escapeHtml(text) {
  if (text === null || text === undefined) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function switchTab(tabId) {
  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
    btn.setAttribute('aria-selected', 'false');
  });

  const clickedBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b =>
    b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabId)
  );
  if (clickedBtn) {
    clickedBtn.classList.add('active');
    clickedBtn.setAttribute('aria-selected', 'true');
  }

  // Update tab panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.remove('active');
  });

  const activePane = document.getElementById(tabId);
  if (activePane) {
    activePane.classList.add('active');
  }
}

function updateWeightValue(slider) {
  const targetId = slider.getAttribute('data-target');
  const label = document.getElementById(targetId);
  if (label) {
    label.textContent = `${slider.value}%`;
  }
  updateWeightsTotal();
}

function updateWeightsTotal() {
  const sliders = document.querySelectorAll('.weight-slider');
  let total = 0;
  sliders.forEach(s => {
    total += parseInt(s.value, 10) || 0;
  });

  const display = document.getElementById('weightsTotalDisplay');
  if (display) {
    display.textContent = `Total: ${total}% (Normalized)`;
    if (total !== 100) {
      display.style.color = 'var(--warning)';
    } else {
      display.style.color = 'var(--ink-500)';
    }
  }
}

function addCandidateCard(data = null) {
  const container = document.getElementById('candidatesList');
  if (!container) return;

  const currentCount = container.querySelectorAll('.candidate-entry-card').length;
  const newIndex = currentCount + 1;

  const card = document.createElement('div');
  card.className = 'candidate-entry-card';
  card.id = `candidate-card-${newIndex}`;

  const nameVal = data?.name || '';
  const qualVal = data?.qualifications || '';
  const groupVal = data?.audit_group || (newIndex % 2 === 0 ? 'Group B' : 'Group A');

  card.innerHTML = `
    <div class="entry-header">
      <span class="entry-index">Candidate ${newIndex} ${data ? '<span class="badge badge-success" style="font-size:10px;">Parsed</span>' : ''}</span>
      <button type="button" class="btn-remove" onclick="removeCandidateCard(this)" title="Remove candidate" aria-label="Remove candidate">&times;</button>
    </div>
    <div class="entry-grid">
      <div>
        <label class="form-label">Candidate Name</label>
        <input type="text" name="candidate_name[]" value="${escapeHtml(nameVal)}" class="form-input" placeholder="e.g. Candidate ${newIndex}">
      </div>
      <div>
        <label class="form-label">Audit Group <small class="text-muted">(Demographic audit only)</small></label>
        <select name="audit_group[]" class="form-select">
          <option value="Group A" ${groupVal === 'Group A' ? 'selected' : ''}>Group A</option>
          <option value="Group B" ${groupVal === 'Group B' ? 'selected' : ''}>Group B</option>
          <option value="Cohort A" ${groupVal === 'Cohort A' ? 'selected' : ''}>Cohort A</option>
          <option value="Cohort B" ${groupVal === 'Cohort B' ? 'selected' : ''}>Cohort B</option>
          <option value="Cohort C" ${groupVal === 'Cohort C' ? 'selected' : ''}>Cohort C</option>
          <option value="Not specified" ${groupVal === 'Not specified' ? 'selected' : ''}>Not specified</option>
        </select>
      </div>
    </div>
    <div class="mt-2">
      <label class="form-label">Qualifications, Skills &amp; Experience</label>
      <textarea name="qualifications[]" class="form-textarea form-textarea-sm" rows="4" placeholder="Technical skills, employment duration, degrees...">${escapeHtml(qualVal)}</textarea>
    </div>
  `;

  container.appendChild(card);
  return card;
}

function removeCandidateCard(btn) {
  const card = btn.closest('.candidate-entry-card');
  const container = document.getElementById('candidatesList');
  if (card && container) {
    const total = container.querySelectorAll('.candidate-entry-card').length;
    if (total <= 2) {
      alert("A minimum of two candidates is required for ranking evaluation.");
      return;
    }
    card.remove();
    // Re-index remaining cards
    container.querySelectorAll('.candidate-entry-card').forEach((el, idx) => {
      const idxSpan = el.querySelector('.entry-index');
      if (idxSpan) {
        idxSpan.textContent = `Candidate ${idx + 1}`;
      }
    });
  }
}

// Asynchronously parse uploaded resume files via /api/parse-resume
async function parseResumeFileAsync(file, pillElement) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/parse-resume', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (data.status === 'success') {
      pillElement.classList.add('parsed-success');
      pillElement.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"></path></svg>
        <span><strong>${escapeHtml(data.candidate_name)}</strong> (${escapeHtml(file.name)})</span>
        <span class="badge badge-success" style="font-size:10px;">${data.skills.length} skills identified</span>
      `;

      // Check if there is an empty candidate form card to populate with metadata
      const emptyCard = Array.from(document.querySelectorAll('.candidate-entry-card')).find(card => {
        const nameInput = card.querySelector('input[name="candidate_name[]"]');
        const qualArea = card.querySelector('textarea[name="qualifications[]"]');
        return (!nameInput || !nameInput.value.trim()) && (!qualArea || !qualArea.value.trim());
      });

      if (emptyCard) {
        const nameInput = emptyCard.querySelector('input[name="candidate_name[]"]');
        const qualArea = emptyCard.querySelector('textarea[name="qualifications[]"]');
        if (nameInput) nameInput.value = data.candidate_name;
        if (qualArea && data.text_preview) {
          qualArea.value = `[Uploaded File: ${file.name}]\n` + data.text_preview;
        }
      }
    } else {
      pillElement.classList.add('parsed-error');
      pillElement.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <span>${escapeHtml(file.name)}: ${escapeHtml(data.message)}</span>
      `;
    }
  } catch (err) {
    pillElement.classList.add('parsed-error');
    pillElement.innerHTML = `<span>${escapeHtml(file.name)} (upload error)</span>`;
  }
}

// Interactive Pairwise Comparison
async function executePairwiseComparison() {
  const selA = document.getElementById('pairwiseCandA');
  const selB = document.getElementById('pairwiseCandB');
  const resultsContainer = document.getElementById('pairwiseResultsArea');

  if (!selA || !selB || !resultsContainer) return;

  const idA = selA.value;
  const idB = selB.value;

  if (!idA || !idB) {
    alert("Please select both Candidate A and Candidate B for pairwise comparison.");
    return;
  }

  if (idA === idB) {
    alert("Please select two distinct candidates to generate a pairwise justification.");
    return;
  }

  resultsContainer.innerHTML = `
    <div style="text-align: center; padding: 24px; color: var(--ink-500);">
      <p>Computing multi-aspect pairwise justification...</p>
    </div>
  `;

  try {
    const response = await fetch('/api/pairwise-compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidate_a_id: idA,
        candidate_b_id: idB,
      }),
    });

    const data = await response.json();
    if (data.status === 'success' && data.comparison) {
      renderPairwiseResults(data.comparison, resultsContainer);
    } else {
      resultsContainer.innerHTML = `<div class="alert alert-error">Error: ${escapeHtml(data.message || 'Comparison failed.')}</div>`;
    }
  } catch (err) {
    resultsContainer.innerHTML = `<div class="alert alert-error">Failed to connect to comparison API endpoint.</div>`;
  }
}

function renderPairwiseResults(comp, container) {
  const diffs = comp.aspect_differences || {};
  const uniqueA = comp.skills_unique_to_a || [];
  const uniqueB = comp.skills_unique_to_b || [];
  const shared = comp.shared_skills || [];

  container.innerHTML = `
    <div class="pairwise-results-view">
      <div class="pairwise-verdict-banner">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        <div>
          <h4 style="margin: 0 0 4px 0; font-size: 15px; color: var(--ink-900);"><strong>Comparative Rationale:</strong> ${escapeHtml(comp.higher_candidate)} Ranked Higher</h4>
          <p style="margin: 0; font-size: 13px; color: var(--ink-700);">${escapeHtml(comp.verdict_text)}</p>
        </div>
      </div>

      <table class="pairwise-aspects-table">
        <thead>
          <tr>
            <th>Evaluation Dimension</th>
            <th>Candidate A (${escapeHtml(comp.candidate_a_name)})</th>
            <th>Candidate B (${escapeHtml(comp.candidate_b_name)})</th>
            <th>Advantage / Delta</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Skill Match Score</strong></td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>
              <strong class="${diffs.skill_match_diff >= 0 ? 'text-success' : 'text-danger'}">
                ${diffs.skill_match_diff >= 0 ? '+' : ''}${diffs.skill_match_diff} pts
              </strong>
            </td>
          </tr>
          <tr>
            <td><strong>Semantic Relevance</strong></td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>
              <strong class="${diffs.semantic_diff >= 0 ? 'text-success' : 'text-danger'}">
                ${diffs.semantic_diff >= 0 ? '+' : ''}${diffs.semantic_diff} pts
              </strong>
            </td>
          </tr>
          <tr>
            <td><strong>Experience Alignment</strong></td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>
              <strong class="${diffs.experience_diff >= 0 ? 'text-success' : 'text-danger'}">
                ${diffs.experience_diff >= 0 ? '+' : ''}${diffs.experience_diff} pts
              </strong>
            </td>
          </tr>
          <tr>
            <td><strong>Education Fit</strong></td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>
              <strong class="${diffs.education_diff >= 0 ? 'text-success' : 'text-danger'}">
                ${diffs.education_diff >= 0 ? '+' : ''}${diffs.education_diff} pts
              </strong>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="mt-4">
        <label class="form-label"><strong>Skill Attribution Breakdown:</strong></label>
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 6px;">
          ${uniqueA.length > 0 ? `
            <div>
              <span class="text-muted" style="font-size: 12px;">Unique to ${escapeHtml(comp.candidate_a_name)}:</span>
              <div class="skill-chips-container mt-1">
                ${uniqueA.map(s => `<span class="skill-chip chip-bonus">${escapeHtml(s)}</span>`).join('')}
              </div>
            </div>
          ` : ''}

          ${uniqueB.length > 0 ? `
            <div>
              <span class="text-muted" style="font-size: 12px;">Unique to ${escapeHtml(comp.candidate_b_name)}:</span>
              <div class="skill-chips-container mt-1">
                ${uniqueB.map(s => `<span class="skill-chip chip-bonus">${escapeHtml(s)}</span>`).join('')}
              </div>
            </div>
          ` : ''}

          ${shared.length > 0 ? `
            <div>
              <span class="text-muted" style="font-size: 12px;">Shared Matched Skills:</span>
              <div class="skill-chips-container mt-1">
                ${shared.map(s => `<span class="skill-chip chip-matched">&check; ${escapeHtml(s)}</span>`).join('')}
              </div>
            </div>
          ` : ''}
        </div>
      </div>
    </div>
  `;
}

// File Drag & Drop Setup
document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileList = document.getElementById('fileList');

  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      fileInput.files = files;
      processFiles(files);
    });

    fileInput.addEventListener('change', () => {
      processFiles(fileInput.files);
    });
  }

  function processFiles(files) {
    if (!fileList || !files.length) return;
    fileList.innerHTML = '';

    Array.from(files).forEach((file) => {
      const pill = document.createElement('div');
      pill.className = 'file-pill';
      pill.style.cssText = 'padding: 8px 12px; background: var(--surface-alt); border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: 6px; font-size: 13px; display: flex; align-items: center; gap: 8px;';
      pill.innerHTML = `
        <span>📄 ${escapeHtml(file.name)}</span>
        <small class="text-muted">(${(file.size / 1024).toFixed(1)} KB)</small>
      `;
      fileList.appendChild(pill);
      parseResumeFileAsync(file, pill);
    });
  }

  updateWeightsTotal();
});
