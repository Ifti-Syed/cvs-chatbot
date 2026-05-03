'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_BASE         = '';
const MAX_HISTORY_SEND = 10;
const MAX_SESSIONS     = 25;
const STORAGE_KEY      = 'cvs_sessions';
const SESSION_KEY      = 'cvs_session_id';
const THEME_KEY        = 'cvs_theme';
const AUTH_KEY         = 'cvs_auth';   // { access_token, refresh_token, user }

// ---------------------------------------------------------------------------
// AuthManager — handles login, logout, token storage, auto-refresh
// ---------------------------------------------------------------------------
const Auth = {
  _data: null,

  load() {
    try {
      const raw = localStorage.getItem(AUTH_KEY);
      this._data = raw ? JSON.parse(raw) : null;
    } catch { this._data = null; }
    return this._data;
  },

  save(tokens, user) {
    this._data = { ...tokens, user };
    localStorage.setItem(AUTH_KEY, JSON.stringify(this._data));
  },

  clear() {
    this._data = null;
    localStorage.removeItem(AUTH_KEY);
  },

  get token()        { return this._data?.access_token  || null; },
  get refreshToken() { return this._data?.refresh_token || null; },
  get user()         { return this._data?.user          || null; },
  get isLoggedIn()   { return !!this.token; },

  headers() {
    return this.token
      ? { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.token}` }
      : { 'Content-Type': 'application/json' };
  },

  async tryRefresh() {
    if (!this.refreshToken) return false;
    try {
      const res = await fetch(`${API_BASE}/api/auth/refresh`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ refresh_token: this.refreshToken }),
      });
      if (!res.ok) return false;
      const tokens = await res.json();
      // Keep existing user object; update tokens
      this.save(tokens, this.user);
      return true;
    } catch { return false; }
  },

  async fetchMe() {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
      headers: this.headers(),
    });
    if (!res.ok) return null;
    return res.json();
  },
};

// ---------------------------------------------------------------------------
// ChatApp
// ---------------------------------------------------------------------------
class ChatApp {
  constructor() {
    this.messages          = [];
    this.isLoading         = false;
    this.currentSession    = null;
    this.sessions          = {};
    this.sidebarOpen       = window.innerWidth > 768;
    this.theme             = localStorage.getItem(THEME_KEY) || 'light';
    this.dom               = {};
    this._activeMenu       = null; // currently open { id, el } context menu
    this._activeController = null; // AbortController for the in-flight stream request
  }

  // =========================================================================
  // Boot
  // =========================================================================

  async init() {
    this._cacheDOM();
    this._applyTheme();
    this._initLightbox();
    // Auth disabled — go straight to the chat UI
    this._showApp({ full_name: 'Guest', email: '', department: '', role: '' });
  }

  // ── Auth helpers ──────────────────────────────────────────────────────────

  _showLogin() {
    document.getElementById('loginOverlay').style.display = 'flex';
    document.getElementById('appShell').style.display     = 'none';
    // Apply theme to login page too
    this._applyTheme();
  }

  _showApp(user) {
    document.getElementById('loginOverlay').style.display = 'none';
    document.getElementById('appShell').style.display     = '';
    this._updateUserBar(user);
    this._bindEvents();
    this._loadStorage();
    this._renderHistory();
    if (this.currentSession && this.sessions[this.currentSession]) {
      this._restoreSession(this.currentSession);
    } else {
      this._newSession();
    }
  }

  _updateUserBar(user) {
    const nameEl  = document.getElementById('userName');
    const deptEl  = document.getElementById('userDept');
    const avatarEl = document.getElementById('userAvatar');
    if (!user) return;
    const first = (user.full_name || user.email || '?').charAt(0).toUpperCase();
    if (avatarEl) avatarEl.textContent = first;
    if (nameEl)   nameEl.textContent   = user.full_name || user.email;
    if (deptEl)   deptEl.textContent   = user.department ? `${user.department} · ${user.role}` : user.role;
  }

  _bindLoginEvents() {
    const form       = document.getElementById('loginForm');
    const btnLogin   = document.getElementById('btnLogin');
    const errorEl    = document.getElementById('loginError');
    const emailEl    = document.getElementById('loginEmail');
    const pwEl       = document.getElementById('loginPassword');
    const togglePwEl = document.getElementById('btnTogglePw');

    // Password visibility toggle
    if (togglePwEl) {
      togglePwEl.addEventListener('click', () => {
        const isHidden = pwEl.type === 'password';
        pwEl.type = isHidden ? 'text' : 'password';
        togglePwEl.querySelector('.eye-open').style.display = isHidden ? 'none' : '';
        togglePwEl.querySelector('.eye-shut').style.display = isHidden ? '' : 'none';
        togglePwEl.setAttribute('aria-label', isHidden ? 'Hide password' : 'Show password');
      });
    }

    const setError = (msg) => {
      errorEl.textContent = msg;
      errorEl.classList.toggle('visible', !!msg);
    };

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      setError('');
      const email    = emailEl.value.trim();
      const password = pwEl.value;
      if (!email || !password) { setError('Please enter your email and password.'); return; }

      btnLogin.disabled = true;
      btnLogin.classList.add('loading');

      try {
        const res = await fetch(`${API_BASE}/api/auth/login`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ email, password }),
        });

        const data = await res.json();

        if (!res.ok) {
          setError(data.detail || 'Login failed. Check your credentials.');
          return;
        }

        // Fetch user profile
        Auth.save(data, null);
        const me = await Auth.fetchMe();
        Auth.save(data, me);

        this._showApp(me);

      } catch (err) {
        setError('Could not reach the server. Please try again.');
      } finally {
        btnLogin.disabled = false;
        btnLogin.classList.remove('loading');
      }
    });
  }

  _cacheDOM() {
    const q = (id) => document.getElementById(id);
    this.dom = {
      appShell:          q('appShell'),
      sidebar:           q('sidebar'),
      btnNewChat:        q('btnNewChat'),
      btnToggle:         q('btnToggleSidebar'),
      btnTheme:          q('btnTheme'),
      chatHistory:       q('chatHistory'),
      historyEmpty:      q('historyEmpty'),
      messagesContainer: q('messagesContainer'),
      welcomeState:      q('welcomeState'),
      typingIndicator:   q('typingIndicator'),
      messageInput:      q('messageInput'),
      btnSend:           q('btnSend'),
      toastContainer:    q('toastContainer'),
    };
  }

  _bindEvents() {
    const { dom } = this;

    dom.btnTheme.addEventListener('click',   () => this._toggleTheme());
    dom.btnNewChat.addEventListener('click', () => this._newSession());
    dom.btnToggle.addEventListener('click',  () => this._toggleSidebar());

    // Logout button hidden (auth disabled)
    const btnLogout = document.getElementById('btnLogout');
    if (btnLogout) btnLogout.style.display = 'none';

    dom.messageInput.addEventListener('input',   () => this._onInputChange());
    dom.messageInput.addEventListener('keydown', (e) => this._onKeyDown(e));
    dom.btnSend.addEventListener('click', () => this.send());

    // Suggestion chips
    dom.messagesContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.suggestion-chip');
      if (!chip) return;
      const msg = chip.dataset.msg;
      if (msg) { dom.messageInput.value = msg; this._onInputChange(); this.send(); }
    });

    // Close context menu on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.history-context-menu')) this._closeMenu();
    });

    // Keyboard: Escape closes sidebar on mobile
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this._closeMenu();
        if (window.innerWidth <= 768 && this.sidebarOpen) this._toggleSidebar();
      }
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) { this.sidebarOpen = true; this._syncSidebar(); }
    });
  }

  // =========================================================================
  // Sessions
  // =========================================================================

  _abortActiveRequest() {
    if (this._activeController) {
      this._activeController.abort();
      this._activeController = null;
      // _showTyping(false) will be called by send()'s finally block once the
      // abort propagates — no need to call it here.
    }
  }

  _newSession() {
    this._abortActiveRequest();
    this.currentSession = `s_${Date.now()}`;
    this.messages = [];
    this._clearMessages();
    this._showWelcome(true);
    this._persist();
    this._renderHistory();
    localStorage.setItem(SESSION_KEY, this.currentSession);
  }

  _switchSession(id) {
    this._abortActiveRequest();
    this.currentSession = id;
    this.messages = this.sessions[id]?.messages || [];
    localStorage.setItem(SESSION_KEY, id);
    this._clearMessages();
    this._restoreSession(id);
    this._renderHistory();
    if (window.innerWidth <= 768) this._toggleSidebar();
  }

  _restoreSession(id) {
    const s = this.sessions[id];
    if (!s?.messages?.length) { this._showWelcome(true); return; }
    this.messages = s.messages;
    this._showWelcome(false);
    this.messages.forEach((m) => this._renderBubble(m.role, m.content, m.sources, m.error));
    this.scrollToBottom(false);
  }

  // =========================================================================
  // Sending
  // =========================================================================

  async send() {
    const text = this.dom.messageInput.value.trim();
    if (!text || this.isLoading) return;

    // Capture history BEFORE adding the current message so the backend
    // receives only prior turns. The backend appends the current message
    // itself — sending it in history too would cause a duplicate.
    const history = this._buildHistory();

    this.dom.messageInput.value = '';
    this._onInputChange();
    this._showWelcome(false);

    this._addMsg('user', text);
    this.scrollToBottom();
    this._showTyping(true);

    try {
      await this._callChatStream(text, history);
    } catch (err) {
      // AbortError means the user started a new chat — don't show an error.
      if (err.name !== 'AbortError') {
        this._addMsg('assistant', null, [], err.message || 'Something went wrong.');
        this._toast('Could not reach the server. Please try again.', 'error');
      }
    } finally {
      // Always unlock the input once the response stream is fully complete.
      this._showTyping(false);
    }

    this.scrollToBottom();
    this._persist();
    this._renderHistory();
  }

  _addMsg(role, content, sources = [], error = null) {
    this.messages.push({ role, content: content || '', sources, error, ts: new Date().toISOString() });
    this._renderBubble(role, content, sources, error);
  }

  // =========================================================================
  // API
  // =========================================================================

  async _callChatStream(message, history) {
    // Transient network errors (ERR_NAME_NOT_RESOLVED, ERR_QUIC_PROTOCOL_ERROR)
    // are retried once after a short pause before surfacing the error to the user.
    const MAX_ATTEMPTS = 2;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        return await this._attemptChatStream(message, history);
      } catch (err) {
        const isTransient = err.name === 'TypeError' || err.message?.includes('timed out');
        if (attempt < MAX_ATTEMPTS && isTransient) {
          await new Promise((r) => setTimeout(r, 1500));
          continue;
        }
        throw err;
      }
    }
  }

  async _attemptChatStream(message, history) {
    // AbortController lets us cancel the fetch on timeout or when the user
    // starts a new chat / switches sessions.
    const controller = new AbortController();
    this._activeController = controller;
    const timeoutId  = setTimeout(() => controller.abort(), 45000); // 45 s hard limit

    let res;
    try {
      res = await fetch(`${API_BASE}/api/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message, conversation_history: history, stream: true }),
        signal:  controller.signal,
      });
    } catch (err) {
      clearTimeout(timeoutId);
      this._activeController = null;
      if (err.name === 'AbortError') return; // silently discard — user navigated away
      throw err;
    }

    if (!res.ok) {
      clearTimeout(timeoutId);
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const reader   = res.body.getReader();
    const decoder  = new TextDecoder();
    let   bubble   = null;
    let   fullText = '';
    let   sources  = [];
    let   buffer   = '';   // accumulates raw bytes until a complete SSE event arrives
    let   done_    = false;

    // Hide the spinner — streaming has started.  isLoading stays true until
    // the full response is done (controlled by the finally block in send()).
    this._hideTypingIndicator();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Append decoded bytes to buffer; stream:true handles multi-byte chars
      buffer += decoder.decode(value, { stream: true });

      // SSE events are delimited by \n\n — split and keep the trailing fragment
      const events = buffer.split('\n\n');
      buffer = events.pop(); // incomplete event tail stays in buffer

      for (const event of events) {
        // Each event may have multiple lines; find the data: line
        const dataLine = event.split('\n').find((l) => l.startsWith('data: '));
        if (!dataLine) continue;

        let msg;
        try { msg = JSON.parse(dataLine.slice(6)); } catch { continue; }

        if (msg.type === 'sources') {
          sources = msg.sources || [];

        } else if (msg.type === 'token') {
          fullText += msg.content;
          if (!bubble) {
            // Create the assistant bubble on the first token
            this.messages.push({ role: 'assistant', content: '', sources, ts: new Date().toISOString() });
            this._renderBubble('assistant', '', sources);
            const rows = this.dom.messagesContainer.querySelectorAll('.message-row.assistant');
            bubble = rows[rows.length - 1].querySelector('.message-bubble');
          }
          if (bubble) {
            bubble.innerHTML = this._format(fullText);
            this.scrollToBottom(false);
          }

        } else if (msg.type === 'done') {
          done_ = true;
          // Remove the streaming bubble and re-render with full content + actions
          if (bubble) {
            const row = bubble.closest('.message-row');
            if (row) { row.remove(); this.messages.pop(); }
          }
          this._addMsg('assistant', fullText || '(No response)', sources);

        } else if (msg.type === 'error') {
          throw new Error(msg.detail || 'Stream error');
        }
      }
    }

    clearTimeout(timeoutId);
    this._activeController = null;

    // Safety net: stream ended without a done event
    if (!done_) {
      // Try to parse leftover buffer as plain JSON (server returned non-streaming response)
      const trimmed = buffer.trim();
      if (trimmed.startsWith('{')) {
        try {
          const json = JSON.parse(trimmed);
          if (json.answer) {
            if (bubble) {
              const row = bubble.closest('.message-row');
              if (row) { row.remove(); this.messages.pop(); }
            }
            this._addMsg('assistant', json.answer, json.sources || []);
            return;
          }
        } catch { /* not valid JSON, fall through */ }
      }
      if (fullText) {
        if (bubble) {
          const row = bubble.closest('.message-row');
          if (row) { row.remove(); this.messages.pop(); }
        }
        this._addMsg('assistant', fullText, sources);
      } else {
        console.error('[CVS] Stream closed without any content. Buffer remainder:', buffer);
        this._addMsg('assistant', null, [], 'No response received. Please try again.');
        this._toast('Stream closed without a response.', 'error');
      }
    }
  }

  _buildHistory() {
    return this.messages.slice(-MAX_HISTORY_SEND).map(({ role, content }) => ({ role, content }));
  }

  // =========================================================================
  // Rendering — bubbles
  // =========================================================================

  _renderBubble(role, content, sources = [], error = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (role === 'assistant') {
      avatar.innerHTML = `
        <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <circle cx="10" cy="10" r="2" fill="#1863DC"/>
          <path d="M10 8C10 5.5 7.5 4 6 5.5C7.5 7 8 8 10 8Z" fill="#1863DC"/>
          <path d="M12 10C14.5 10 16 7.5 14.5 6C13 7.5 12 8 12 10Z" fill="#1863DC" opacity="0.75"/>
          <path d="M10 12C10 14.5 12.5 16 14 14.5C12.5 13 12 12 10 12Z" fill="#1863DC" opacity="0.55"/>
          <path d="M8 10C5.5 10 4 12.5 5.5 14C7 12.5 8 12 8 10Z" fill="#1863DC" opacity="0.38"/>
        </svg>`;
    } else {
      avatar.textContent = 'You';
    }

    // Content wrapper
    const wrap = document.createElement('div');
    wrap.className = 'message-content-wrapper';

    if (error) {
      const el = document.createElement('div');
      el.className = 'message-error';
      el.textContent = `⚠ ${error}`;
      wrap.appendChild(el);
    } else {
      const bubble = document.createElement('div');
      bubble.className = 'message-bubble';
      bubble.innerHTML = this._format(content || '');
      wrap.appendChild(bubble);
    }

    // Show inline diagram images only when source_type === 'image_caption'
    // (user asked for a specific figure). PDF text and table answers never
    // render images — the citation at the end of the answer is sufficient.
    if (sources?.length) {
      const imageSources = sources.filter(
        (s) => s.source_type === 'image_caption' && s.image_blob_names?.length,
      );
      if (imageSources.length) {
        const srcWrap = document.createElement('div');
        srcWrap.className = 'message-sources';

        imageSources.forEach((s) => {
          s.image_blob_names.forEach((blobName) => {
            const parts   = blobName.split('/');
            const docPart = parts[1] || s.document_name;
            const filePart = parts[parts.length - 1];
            const imgUrl  = `/api/images/${encodeURIComponent(docPart)}/${encodeURIComponent(filePart)}`;
            const label   = s.figure_name || s.document_name;

            const imgWrap = document.createElement('div');
            imgWrap.className = 'source-image-wrap';

            const badge = document.createElement('span');
            badge.className = 'figure-badge';
            badge.innerHTML = `
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                <rect x="1" y="2" width="14" height="11" rx="1.5"/>
                <path d="M1 10l4-4 3 3 2-2 5 4"/>
              </svg>
              ${this._esc(label)}`;

            const img = document.createElement('img');
            img.src       = imgUrl;
            img.className = 'source-image';
            img.loading   = 'lazy';
            img.alt       = label;
            img.title     = 'Click to expand';
            img.addEventListener('click', () => this._openLightbox(imgUrl, label));

            imgWrap.appendChild(badge);
            imgWrap.appendChild(img);
            srcWrap.appendChild(imgWrap);
          });
        });

        wrap.appendChild(srcWrap);
      }
    }

    // Edit + Copy buttons — user messages
    if (role === 'user' && content) {
      const actions = document.createElement('div');
      actions.className = 'message-actions user-actions';

      const btnCopy = document.createElement('button');
      btnCopy.className = 'btn-msg-action';
      btnCopy.title = 'Copy message';
      btnCopy.setAttribute('aria-label', 'Copy message');
      btnCopy.innerHTML = `
        <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        <svg class="icon-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
          <polyline points="20 6 9 17 4 12"/>
        </svg>`;
      btnCopy.addEventListener('click', () => this._copyResponse(btnCopy, content));

      const btnEdit = document.createElement('button');
      btnEdit.className = 'btn-msg-action';
      btnEdit.title = 'Edit message';
      btnEdit.setAttribute('aria-label', 'Edit message');
      btnEdit.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
        </svg>`;
      btnEdit.addEventListener('click', () => this._editUserMsg(row, content));

      actions.appendChild(btnCopy);
      actions.appendChild(btnEdit);
      wrap.appendChild(actions);
    }

    // Copy button — assistant responses only
    if (role === 'assistant' && !error && content) {
      const actions = document.createElement('div');
      actions.className = 'message-actions';
      const btnCopy = document.createElement('button');
      btnCopy.className = 'btn-copy-response';
      btnCopy.title = 'Copy response';
      btnCopy.setAttribute('aria-label', 'Copy response');
      btnCopy.innerHTML = `
        <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        <svg class="icon-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
          <polyline points="20 6 9 17 4 12"/>
        </svg>`;
      btnCopy.addEventListener('click', () => this._copyResponse(btnCopy, content));
      actions.appendChild(btnCopy);
      wrap.appendChild(actions);
    }

    // Timestamp
    const ts = document.createElement('div');
    ts.className = 'message-timestamp';
    ts.textContent = this._time(new Date());
    wrap.appendChild(ts);

    row.appendChild(avatar);
    row.appendChild(wrap);
    this.dom.messagesContainer.appendChild(row);
  }

  _copyResponse(btn, text) {
    navigator.clipboard.writeText(text)
      .then(() => {
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 2000);
      })
      .catch(() => this._toast('Could not copy to clipboard', 'error'));
  }

  _editUserMsg(row, content) {
    if (row.classList.contains('editing')) return;
    row.classList.add('editing');

    const wrap   = row.querySelector('.message-content-wrapper');
    const bubble = row.querySelector('.message-bubble');

    // Swap bubble for a resizable textarea
    const ta = document.createElement('textarea');
    ta.className = 'edit-msg-textarea';
    ta.value = content;
    bubble.replaceWith(ta);

    // Auto-height
    const resize = () => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 300) + 'px'; };
    resize();
    ta.addEventListener('input', resize);

    // Send / Cancel buttons
    const btnRow = document.createElement('div');
    btnRow.className = 'edit-msg-buttons';

    const btnCancel = document.createElement('button');
    btnCancel.className = 'btn-edit-cancel';
    btnCancel.textContent = 'Cancel';
    btnCancel.addEventListener('click', () => {
      ta.replaceWith(bubble);
      btnRow.remove();
      row.classList.remove('editing');
    });

    const btnSend = document.createElement('button');
    btnSend.className = 'btn-edit-send';
    btnSend.textContent = 'Send';
    btnSend.addEventListener('click', () => {
      const newText = ta.value.trim();
      if (newText && !this.isLoading) this._submitEditedMsg(row, newText);
    });

    // Enter sends, Escape cancels
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const newText = ta.value.trim();
        if (newText && !this.isLoading) this._submitEditedMsg(row, newText);
      }
      if (e.key === 'Escape') btnCancel.click();
    });

    btnRow.appendChild(btnCancel);
    btnRow.appendChild(btnSend);
    wrap.appendChild(btnRow);

    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  }

  _submitEditedMsg(row, newText) {
    const allRows  = Array.from(this.dom.messagesContainer.querySelectorAll('.message-row'));
    const rowIndex = allRows.indexOf(row);

    // Remove this row and everything that came after it
    allRows.slice(rowIndex).forEach((r) => r.remove());

    // Trim messages array to match
    this.messages.splice(rowIndex);

    // Feed the edited text into the normal send flow
    this.dom.messageInput.value = newText;
    this._onInputChange();
    this._showWelcome(false);
    this.send();
  }

  _clearMessages() {
    this.dom.messagesContainer.querySelectorAll('.message-row').forEach((el) => el.remove());
  }

  _showWelcome(show) {
    this.dom.welcomeState.style.display = show ? '' : 'none';
  }

  // =========================================================================
  // History sidebar
  // =========================================================================

  _renderHistory() {
    const { chatHistory, historyEmpty } = this.dom;
    chatHistory.innerHTML = '';

    // Sort: pinned first, then newest first
    const ids = Object.keys(this.sessions)
      .filter((id) => this.sessions[id].messages?.some((m) => m.role === 'user'))
      .sort((a, b) => {
        const sa = this.sessions[a], sb = this.sessions[b];
        if (sa.pinned && !sb.pinned) return -1;
        if (!sa.pinned && sb.pinned) return 1;
        return (sb.ts || 0) - (sa.ts || 0);
      })
      .slice(0, 20);

    if (!ids.length) {
      chatHistory.appendChild(historyEmpty);
      return;
    }

    ids.forEach((id) => {
      const s     = this.sessions[id];
      const label = s.title || s.messages?.find((m) => m.role === 'user')?.content || 'New Chat';

      const item = document.createElement('div');
      item.className = `history-item${id === this.currentSession ? ' active' : ''}${s.pinned ? ' pinned' : ''}`;
      item.dataset.sessionId = id;

      // Chat icon
      item.innerHTML = `
        <svg class="chat-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
        </svg>
        ${s.pinned ? `<svg class="pin-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/></svg>` : ''}
        <span class="history-label">${this._esc(this._trunc(label, 22))}</span>
        <button class="btn-history-menu" aria-label="Chat options" title="Options">
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <circle cx="12" cy="5"  r="1.4"/>
            <circle cx="12" cy="12" r="1.4"/>
            <circle cx="12" cy="19" r="1.4"/>
          </svg>
        </button>`;

      // Click on item body → switch session
      item.addEventListener('click', (e) => {
        if (e.target.closest('.btn-history-menu')) return;
        this._switchSession(id);
      });

      // Click ⋮ → open context menu
      item.querySelector('.btn-history-menu').addEventListener('click', (e) => {
        e.stopPropagation();
        this._openMenu(e.currentTarget, id, s.pinned);
      });

      chatHistory.appendChild(item);
    });
  }

  // =========================================================================
  // Context menu (fixed-positioned, appended to body)
  // =========================================================================

  _openMenu(btn, id, isPinned) {
    this._closeMenu();

    const rect = btn.getBoundingClientRect();
    const menu = document.createElement('div');
    menu.className = 'history-context-menu';

    menu.innerHTML = `
      <button data-action="rename">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
        </svg>
        Rename
      </button>
      <button data-action="pin">
        <svg viewBox="0 0 24 24" fill="${isPinned ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/>
        </svg>
        ${isPinned ? 'Unpin' : 'Pin'}
      </button>
      <button data-action="share">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
          <line x1="15.41" y1="6.51"  x2="8.59"  y2="10.49"/>
        </svg>
        Share
      </button>
      <div class="menu-divider"></div>
      <button data-action="delete" class="danger">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
          <path d="M10 11v6M14 11v6M9 6V4h6v2"/>
        </svg>
        Delete
      </button>`;

    // Position: right edge of menu aligns with right edge of button; drops below
    const menuWidth = 164;
    const left = Math.max(6, rect.right - menuWidth);
    menu.style.top  = `${rect.bottom + 4}px`;
    menu.style.left = `${left}px`;

    menu.addEventListener('click', (e) => {
      e.stopPropagation();
      const el = e.target.closest('[data-action]');
      if (!el) return;
      this._closeMenu();
      switch (el.dataset.action) {
        case 'rename': this._renameSession(id); break;
        case 'pin':    this._pinSession(id);    break;
        case 'share':  this._shareSession(id);  break;
        case 'delete': this._deleteSession(id); break;
      }
    });

    document.body.appendChild(menu);
    // Trigger CSS transition
    requestAnimationFrame(() => menu.classList.add('open'));
    this._activeMenu = menu;
  }

  _closeMenu() {
    if (this._activeMenu) {
      this._activeMenu.remove();
      this._activeMenu = null;
    }
  }

  // =========================================================================
  // Session actions
  // =========================================================================

  _renameSession(id) {
    const item = this.dom.chatHistory.querySelector(`[data-session-id="${id}"]`);
    if (!item) return;
    const labelEl = item.querySelector('.history-label');
    if (!labelEl) return;

    const current = this.sessions[id]?.title
      || this.sessions[id]?.messages?.find((m) => m.role === 'user')?.content
      || '';

    const input = document.createElement('input');
    input.type      = 'text';
    input.className = 'history-rename-input';
    input.value     = current.slice(0, 60);
    input.maxLength = 60;

    labelEl.replaceWith(input);
    input.focus();
    input.select();

    const save = () => {
      const val = input.value.trim();
      if (val) this.sessions[id].title = val;
      this._persist();
      this._renderHistory();
    };

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter')  { e.preventDefault(); save(); }
      if (e.key === 'Escape') { e.stopPropagation(); this._renderHistory(); }
    });
    input.addEventListener('blur', save, { once: true });
  }

  _pinSession(id) {
    this.sessions[id].pinned = !this.sessions[id].pinned;
    this._persist();
    this._renderHistory();
    this._toast(this.sessions[id].pinned ? 'Chat pinned' : 'Chat unpinned', 'success');
  }

  _shareSession(id) {
    const s = this.sessions[id];
    if (!s?.messages?.length) return;

    const title = s.title || s.messages.find((m) => m.role === 'user')?.content || 'Chat';
    const date  = new Date(s.ts).toLocaleDateString([], { dateStyle: 'medium' });

    const lines = [`CVS Assistant – "${title}"`, `Date: ${date}`, ''];
    s.messages.forEach((m) => {
      if (m.role === 'user')      lines.push(`You: ${m.content}`, '');
      else if (m.content)         lines.push(`Assistant: ${m.content}`, '');
    });

    navigator.clipboard.writeText(lines.join('\n'))
      .then(() => this._toast('Conversation copied to clipboard', 'success'))
      .catch(() => this._toast('Could not copy to clipboard', 'error'));
  }

  _deleteSession(id) {
    delete this.sessions[id];
    if (this.currentSession === id) {
      this.messages = [];
      this._clearMessages();
      this._showWelcome(true);
      this.currentSession = null;
    }
    this._persist();
    this._renderHistory();
  }

  // =========================================================================
  // Typing indicator
  // =========================================================================

  _showTyping(show) {
    this.isLoading = show;
    this.dom.btnSend.disabled = show || this.dom.messageInput.value.trim().length === 0;
    this.dom.messageInput.disabled = show; // block input until response is complete

    const ti = this.dom.typingIndicator;
    ti.setAttribute('aria-hidden', show ? 'false' : 'true');

    if (show) {
      this.dom.messagesContainer.appendChild(ti);
      this.scrollToBottom();
    } else {
      const main = document.querySelector('.main-area');
      if (main) main.insertBefore(ti, this.dom.messagesContainer.nextSibling);
      this.dom.messageInput.focus();
    }
  }

  // Hide the animated dots without touching isLoading — used when the first
  // streamed token arrives (spinner no longer needed, but input stays locked
  // until the full response is done).
  _hideTypingIndicator() {
    const ti = this.dom.typingIndicator;
    ti.setAttribute('aria-hidden', 'true');
    const main = document.querySelector('.main-area');
    if (main) main.insertBefore(ti, this.dom.messagesContainer.nextSibling);
  }

  // =========================================================================
  // Input
  // =========================================================================

  _onInputChange() {
    const len = this.dom.messageInput.value.length;
    this.dom.btnSend.disabled = len === 0 || this.isLoading;
    this._autoResize(this.dom.messageInput);
  }

  _onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!this.dom.btnSend.disabled) this.send();
    }
  }

  _autoResize(el) {
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  // =========================================================================
  // Sidebar
  // =========================================================================

  _toggleSidebar() {
    this.sidebarOpen = !this.sidebarOpen;
    this._syncSidebar();
  }

  _syncSidebar() {
    this.dom.sidebar.classList.toggle('open', this.sidebarOpen);
    if (window.innerWidth <= 768) {
      let bd = document.querySelector('.sidebar-backdrop');
      if (!bd) {
        bd = document.createElement('div');
        bd.className = 'sidebar-backdrop';
        bd.addEventListener('click', () => this._toggleSidebar());
        document.body.appendChild(bd);
      }
      bd.classList.toggle('visible', this.sidebarOpen);
    }
  }

  // =========================================================================
  // Persistence
  // =========================================================================

  _persist() {
    if (!this.currentSession) return;

    const existing = this.sessions[this.currentSession] || {};
    this.sessions[this.currentSession] = {
      title:    existing.title  || null,
      pinned:   existing.pinned || false,
      messages: [...this.messages],
      ts:       Date.now(),
    };

    // Keep only the most recent MAX_SESSIONS
    const keys = Object.keys(this.sessions)
      .sort((a, b) => (this.sessions[b].ts || 0) - (this.sessions[a].ts || 0));
    keys.slice(MAX_SESSIONS).forEach((k) => delete this.sessions[k]);

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.sessions));
      localStorage.setItem(SESSION_KEY, this.currentSession);
    } catch { /* quota exceeded */ }
  }

  _loadStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) this.sessions = JSON.parse(raw);
      const sid = localStorage.getItem(SESSION_KEY);
      if (sid && this.sessions[sid]) this.currentSession = sid;
    } catch { this.sessions = {}; }
  }

  // =========================================================================
  // Theme
  // =========================================================================

  _toggleTheme() {
    this.theme = this.theme === 'light' ? 'dark' : 'light';
    this._applyTheme();
    localStorage.setItem(THEME_KEY, this.theme);
  }

  _applyTheme() {
    const isDark = this.theme === 'dark';
    document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
    const btn = (this.dom && this.dom.btnTheme) || document.getElementById('btnTheme');
    if (!btn) return;
    btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    btn.setAttribute('title',      isDark ? 'Switch to light mode' : 'Switch to dark mode');
  }

  // =========================================================================
  // Markdown-light formatter
  // =========================================================================

  _format(text) {
    if (!text) return '';
    let h = this._esc(text);
    h = h.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
    h = h.replace(/`([^`\n]+?)`/g, '<code>$1</code>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
    h = h.replace(/^[ \t]*[-*•] (.+)$/gm, '<li>$1</li>');
    h = h.replace(/(<li>[\s\S]+?<\/li>)/g, '<ul>$1</ul>');
    h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    h = h.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
    if (!h.startsWith('<')) h = `<p>${h}</p>`;
    return h;
  }

  // =========================================================================
  // Toasts
  // =========================================================================

  _toast(msg, type = 'info') {
    const icons = { success: '✓', error: '✗', warning: '⚠', info: 'ℹ' };
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span class="toast-message">${this._esc(msg)}</span>`;
    this.dom.toastContainer.appendChild(el);
    setTimeout(() => el.remove(), 4300);
  }

  // =========================================================================
  // Scroll
  // =========================================================================

  scrollToBottom(smooth = true) {
    const c = this.dom.messagesContainer;
    c.scrollTo({ top: c.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
  }

  // =========================================================================
  // Utilities
  // =========================================================================

  _esc(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
  }

  _trunc(s, n) {
    return s && s.length > n ? s.slice(0, n) + '…' : (s || '');
  }

  _time(d) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // =========================================================================
  // Lightbox — full-screen elastic image viewer
  // =========================================================================

  _initLightbox() {
    const overlay = document.createElement('div');
    overlay.id = 'lightbox';
    overlay.className = 'lightbox';
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-label', 'Image viewer');
    overlay.innerHTML = `
      <div class="lightbox-backdrop"></div>
      <div class="lightbox-content">
        <button class="lightbox-close" aria-label="Close image viewer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
        <img class="lightbox-img" src="" alt="" />
        <div class="lightbox-caption"></div>
      </div>`;

    document.body.appendChild(overlay);

    overlay.querySelector('.lightbox-backdrop').addEventListener('click', () => this._closeLightbox());
    overlay.querySelector('.lightbox-close').addEventListener('click', () => this._closeLightbox());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.classList.contains('open')) this._closeLightbox();
    });

    this._lightbox = overlay;
  }

  _openLightbox(imgUrl, caption) {
    if (!this._lightbox) return;
    this._lightbox.querySelector('.lightbox-img').src = imgUrl;
    this._lightbox.querySelector('.lightbox-img').alt = caption;
    this._lightbox.querySelector('.lightbox-caption').textContent = caption;
    this._lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  _closeLightbox() {
    if (!this._lightbox) return;
    this._lightbox.classList.remove('open');
    document.body.style.overflow = '';
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
const app = new ChatApp();
document.addEventListener('DOMContentLoaded', () => app.init());
