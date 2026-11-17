/**
 * FairHire AI — Frontend Interactivity
 * Handles tab navigation, dynamic candidate management, drag-and-drop resume upload
 */

function switchTab(tabId) {
  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  
  // Find button corresponding to tabId
  const clickedBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => 
    b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabId)
  );
  if (clickedBtn) {
    clickedBtn.classList.add('active');
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

function addCandidateCard() {
  const container = document.getElementById('candidatesList');
  if (!container) return;

  const currentCount = container.querySelectorAll('.candidate-entry-card').length;
  const newIndex = currentCount + 1;

  const card = document.createElement('div');
  card.className = 'candidate-entry-card';
  card.id = `candidate-card-${newIndex}`;

  card.innerHTML = `
    <div class="entry-header">
      <span class="entry-index">Candidate ${newIndex}</span>
      <button type="button" class="btn-remove" onclick="removeCandidateCard(this)" title="Remove candidate">&times;</button>
    </div>
    <div class="entry-grid">
      <div>
        <label class="form-label">Full Name</label>
        <input type="text" name="candidate_name[]" class="form-input" placeholder="e.g. Candidate ${newIndex}">
      </div>
      <div>
        <label class="form-label">Audit Group <small class="text-muted">(For fairness tracking only)</small></label>
        <select name="audit_group[]" class="form-select">
          <option value="Group A">Group A</option>
          <option value="Group B" ${newIndex % 2 === 0 ? 'selected' : ''}>Group B</option>
          <option value="Not specified">Not specified</option>
        </select>
      </div>
    </div>
    <div class="mt-2">
      <label class="form-label">Qualifications, Skills &amp; Experience</label>
      <textarea name="qualifications[]" class="form-textarea form-textarea-sm" rows="4" placeholder="Technical skills, employment history, degrees..."></textarea>
    </div>
  `;

  container.appendChild(card);
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
      renderFileList(files);
    });

    fileInput.addEventListener('change', () => {
      renderFileList(fileInput.files);
    });
  }

  function renderFileList(files) {
    if (!fileList) return;
    fileList.innerHTML = '';
    Array.from(files).forEach((file) => {
      const pill = document.createElement('div');
      pill.className = 'file-pill';
      pill.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
        <span>${file.name}</span>
        <small class="text-muted">(${(file.size / 1024).toFixed(1)} KB)</small>
      `;
      fileList.appendChild(pill);
    });
  }
});
