        (function() {
            // ============ DATA ============
const COMMANDS = [{
                name: '/help',
                icon: '❓',
                desc: 'Show all available commands',
                shortcut: 'h',
                action: () => showHelp()
            }, {
                name: '/clear',
                icon: '🧹',
                desc: 'Clear the terminal screen',
                shortcut: 'c',
                action: () => clearTerminal()
            }, {
                name: '/create',
                icon: '👤',
                desc: 'Register a new NebulonMind user',
                shortcut: 'u',
                action: (args) => createUser(args)
            }, {
                name: '/setup',
                icon: '🔁',
                desc: 'Switch to an existing user',
                shortcut: 'z',
                action: (args) => setupUser(args)
            }, {
                name: '/whoami',
                icon: '🪪',
                desc: 'Show the current user',
                shortcut: 'n',
                action: () => whoami()
            }, {
                name: '/status',
                icon: '🟢',
                desc: 'API + backend + LLM health',
                shortcut: 's',
                action: () => showServerStatus()
            }, {
                name: '/search',
                icon: '🔎',
                desc: 'Recall memories for a query',
                shortcut: 'r',
                action: (args) => searchMemories(args)
            }, {
                name: '/context',
                icon: '📋',
                desc: 'Build bounded context for a query',
                shortcut: 'x',
                action: (args) => showContext(args)
            }, {
                name: '/sessions',
                icon: '💬',
                desc: 'List active agent sessions',
                shortcut: 'g',
                action: () => showSessions()
            }, {
                name: '/llm',
                icon: '🤖',
                desc: 'Show configured LLM provider status',
                shortcut: 'l',
                action: () => showLLMStatus()
            }, {
                name: '/bg',
                icon: '🕰️',
                desc: 'Background scheduler status',
                shortcut: 'b',
                action: () => backgroundStatus()
            }, {
                name: '/eval',
                icon: '📊',
                desc: 'Run the agent evaluation suite',
                shortcut: 'e',
                action: () => runEval()
            }, ];

// ============ STATE ============
// ============ STATE ============
            let messageCount = 0;
            let commandHistory = [];
            let historyIndex = -1;
            let isPaletteOpen = false;
            let paletteSelectedIndex = 0;
            let filteredCommands = [...COMMANDS];
            let theme = 'dark';

            // ============ DOM REFS ============
            const messageArea = document.getElementById('messageArea');
            const terminalInput = document.getElementById('terminalInput');
            const commandPalette = document.getElementById('commandPalette');
            const paletteItems = document.getElementById('paletteItems');
            const terminalContainer = document.getElementById('terminalContainer');
            const inputArea = document.getElementById('inputArea');
            const inputPrompt = document.querySelector('.input-prompt');
            const bgParticles = document.getElementById('bgParticles');

            // ============ NEBULONMIND API ============
            const API_BASE = window.location.origin + '/api/NebulonMind';
            let currentUser = localStorage.getItem('nmd_user') || null;
            let chatHistory = [];
            let currentChatId = null;
            let savedChats = loadSavedChats();

            function loadSavedChats() {
                try {
                    const raw = JSON.parse(localStorage.getItem('nmd_chats') || '[]');
                    return Array.isArray(raw) ? raw : [];
                } catch (_) { return []; }
            }

            function loadChatsFromServer(restoreMode) {
                apiGet('/chats', { user_id: currentUser }).then(body => {
                    const rows = (body.data && body.data.chats) || [];
                    savedChats = rows;
                    persistChats();
                    renderChatHistory();
                    if (restoreMode === 'server' && rows.length) {
                        renderFullChat(rows[0]);
                    } else if (chatHistory.length && !rows.some(c => c.id === currentChatId)) {
                        touchCurrentChat();
                    }
                }).catch(() => { /* offline: keep the local cache */ });
            }

            function persistChats() {
                localStorage.setItem('nmd_chats', JSON.stringify(savedChats));
            }

            function chatsForUser(user) {
                return savedChats.filter(c => (c.user || '') === user);
            }

            function titleFromHistory(hist) {
                const first = hist.find(m => m.role === 'user');
                return first ? first.content.slice(0, 40) : 'Untitled chat';
            }

            function persistActiveChat() {
                localStorage.setItem('nmd_active_chat', JSON.stringify({
                    id: currentChatId,
                    user: currentUser,
                    messages: chatHistory,
                    createdAt: Date.now(),
                }));
            }

            function touchCurrentChat() {
                if (!chatHistory.length) return;
                persistActiveChat();
                let entry = savedChats.find(c => c.id === currentChatId);
                const createdAt = entry ? entry.createdAt : Date.now();
                if (entry) {
                    entry.messages = chatHistory.slice();
                    entry.title = titleFromHistory(chatHistory);
                    entry.user = currentUser;
                } else {
                    savedChats.unshift({
                        id: currentChatId,
                        title: titleFromHistory(chatHistory),
                        user: currentUser,
                        createdAt,
                        messages: chatHistory.slice(),
                    });
                    entry = savedChats[0];
                }
                persistChats();
                renderChatHistory();
                apiPost('/chats', { user_id: currentUser, chat: entry }).catch(() => { });
            }

            function restoreActiveChat() {
                let mode = 'none';
                try {
                    const raw = JSON.parse(localStorage.getItem('nmd_active_chat') || 'null');
                    if (raw && Array.isArray(raw.messages)) {
                        if (raw.user !== currentUser) {
                            mode = 'server';
                        } else {
                            currentChatId = raw.id || Date.now().toString();
                            chatHistory = raw.messages.slice();
                            chatHistory.forEach(m =>
                                addMessage(m.role === 'assistant' ? 'ai' : m.role, m.content,
                                    m.role === 'assistant' ? 'ai' : 'user'));
                            if (raw.messages.length && !savedChats.find(c => c.id === raw.id)) {
                                savedChats.unshift({
                                    id: raw.id,
                                    title: titleFromHistory(raw.messages),
                                    user: raw.user,
                                    createdAt: raw.createdAt || Date.now(),
                                    messages: raw.messages.slice(),
                                });
                                persistChats();
                            }
                            scrollToBottom();
                            mode = 'local';
                        }
                    } else {
                        mode = 'server';
                    }
                } catch (_) { mode = 'server'; }
                return mode;
            }

            function startNewChat() {
                touchCurrentChat();
                currentChatId = Date.now().toString();
                chatHistory = [];
                persistActiveChat();
                messageArea.innerHTML = '';
                messageCount = 0;
                renderChatHistory();
                scrollToBottom();
                addMessage('success', 'New chat started. 💬', 'success');
                updatePrompt();
            }

            function restoreChat(id) {
                const entry = savedChats.find(c => c.id === id);
                if (!entry) {
                    apiGet(`/chats/${encodeURIComponent(id)}`, { user_id: currentUser }).then(body => {
                        const full = (body.data && body.data.chat) || null;
                        if (!full) return;
                        savedChats.unshift(full);
                        persistChats();
                        renderChatHistory();
                        renderFullChat(full);
                    }).catch(err => addMessage('error', `Chat load failed: ${err.message}`, 'error'));
                    return;
                }
                touchCurrentChat();
                renderFullChat(entry);
            }

            function renderFullChat(entry) {
                messageArea.innerHTML = '';
                messageCount = 0;
                chatHistory = (entry.messages || []).slice();
                chatHistory.forEach(m =>
                    addMessage(m.role === 'assistant' ? 'ai' : m.role, m.content,
                        m.role === 'assistant' ? 'ai' : 'user'));
                currentChatId = entry.id;
                persistActiveChat();
                renderChatHistory();
                scrollToBottom();
            }

            function deleteChat(id) {
                if (id === currentChatId) return;
                savedChats = savedChats.filter(c => c.id !== id);
                persistChats();
                renderChatHistory();
                apiRequest('DELETE', `/chats/${encodeURIComponent(id)}`, { params: { user_id: currentUser } })
                    .catch(() => { });
            }

            function resetChatContext() {
                chatHistory = [];
                currentChatId = Date.now().toString();
                persistActiveChat();
                renderChatHistory();
            }

            async function apiRequest(method, path, options = {}) {
                const { body, params } = options;
                let url = API_BASE + path;
                const qs = new URLSearchParams();
                if (params) {
                    Object.entries(params).forEach(([k, v]) => {
                        if (v !== undefined && v !== null) qs.set(k, v);
                    });
                }
                if (qs.toString()) url += '?' + qs.toString();
                const meta = {
                    method,
                    headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
                };
                if (body !== undefined) meta.body = JSON.stringify(body);
                const resp = await fetch(url, meta);
                let json = null;
                try { json = await resp.json(); } catch (_) { /* non-JSON response */ }
                if (!resp.ok || (json && json.success === false)) {
                    const detail = (json && (json.message || json.detail)) || `HTTP ${resp.status}`;
                    throw new Error(detail);
                }
                return json;
            }
            const apiGet = (path, params) => apiRequest('GET', path, { params });
            const apiPost = (path, body, params) => apiRequest('POST', path, { body, params });

            // ============ INITIALIZATION ============
            function resolveDefaultUser() {
                const saved = localStorage.getItem('nmd_user');
                if (saved) { currentUser = saved; return Promise.resolve(); }
                return apiGet('/config/cfg').then(body => {
                    const groups = (body.data && body.data.groups) || [];
                    const bg = groups.find(g => g.id === 'background') || {};
                    const keys = bg.keys || [];
                    const entry = keys.find(k => k.key === 'nmd_background_user');
                    const value = entry && entry.value ? String(entry.value).trim() : '';
                    if (value && !value.startsWith('/')) currentUser = value;
                }).catch(() => { }).finally(() => {
                    if (!currentUser) currentUser = 'nmd_user_01';
                    localStorage.setItem('nmd_user', currentUser);
                });
            }

            function init() {
                applyTheme(localStorage.getItem('nmd_theme') || 'dark', true);
                currentChatId = Date.now().toString();
                createBackgroundParticles();
                showWelcomeMessage();
                resolveDefaultUser().then(() => {
                    refreshApiBadge();
                    const restored = restoreActiveChat();
                    loadChatsFromServer(restored);
                });
                setupEventListeners();
                terminalInput.focus();
                updatePrompt();
            }

            function refreshApiBadge() {
                const badge = document.querySelector('.header-badge');
                if (!badge) return;
                apiGet('/health').then(body => {
                    const d = body.data || {};
                    badge.innerHTML = `<span class="dot"></span>online · ${currentUser} · ${d.backend || ''} backend`;
                    badge.style.color = 'var(--green)';
                }).catch(() => {
                    badge.innerHTML = 'api offline';
                    badge.style.color = 'var(--red)';
                });
            }

            // ============ BACKGROUND PARTICLES ============
            function createBackgroundParticles() {
                const symbols = ['</>', '{}', '[]', '()', '=>', '&&', '||', ';', '~', '*', '#', '0', '1'];
                for (let i = 0; i < 25; i++) {
                    const span = document.createElement('span');
                    span.textContent = symbols[Math.floor(Math.random() * symbols.length)];
                    span.style.left = Math.random() * 95 + '%';
                    span.style.fontSize = (Math.random() * 10 + 8) + 'px';
                    span.style.animationDuration = (Math.random() * 20 + 15) + 's';
                    span.style.animationDelay = (Math.random() * 15) + 's';
                    span.style.opacity = Math.random() * 0.2 + 0.05;
                    bgParticles.appendChild(span);
                }
            }

            // ============ WELCOME MESSAGE ============
            function showWelcomeMessage() {
                const welcomeLines = [
                    { prompt: 'system', text: '╔══════════════════════════════════════════════════╗', type: 'system' },
                    { prompt: 'system',
                    text: '║   Welcome to NebulonMind Terminal v0.1        ║', type: 'system' },
                    { prompt: 'system', text: '║   Type "/" to see all available commands.         ║', type: 'system' },
                    { prompt: 'system',
                    text: '║   Start chatting or use commands to explore.       ║', type: 'system' },
                    { prompt: 'system',
                    text: '╚══════════════════════════════════════════════════╝', type: 'system' },
                    { prompt: 'ai', text: 'Hello! I\'m NebulonMind, your memory-driven AI assistant. Type /help to see what I can do, or just start chatting! 👋',
                        type: 'ai' },
                ];
                welcomeLines.forEach((line, index) => {
                    setTimeout(() => {
                        addMessage(line.prompt, line.text, line.type);
                    }, index * 120);
                });
            }

            // ============ EVENT LISTENERS ============
            function setupEventListeners() {
                // Input events
                terminalInput.addEventListener('input', handleInput);
                terminalInput.addEventListener('keydown', handleKeydown);
                terminalInput.addEventListener('focus', () => updatePrompt());

                // Palette click events (delegation)
                paletteItems.addEventListener('click', (e) => {
                    const item = e.target.closest('.palette-item');
                    if (item) {
                        const index = parseInt(item.dataset.index);
                        selectPaletteCommand(index);
                    }
                });

                // Palette mouse hover
                paletteItems.addEventListener('mouseover', (e) => {
                    const item = e.target.closest('.palette-item');
                    if (item) {
                        paletteSelectedIndex = parseInt(item.dataset.index);
                        updatePaletteHighlight();
                    }
                });

                // Header buttons
                document.getElementById('btnSettings').addEventListener('click', openSettings);
                document.getElementById('btnNewChat').addEventListener('click', startNewChat);
                document.getElementById('btnExpand').addEventListener('click', () => toggleTheme());

                // Settings panel
                document.getElementById('settingsCloseBtn').addEventListener('click', closeSettings);
                document.getElementById('settingsSaveBtn').addEventListener('click', saveSettings);
                document.getElementById('settingsRestartBtn').addEventListener('click', restartService);
                document.getElementById('settingsOverlay').addEventListener('click', (e) => {
                    if (e.target.id === 'settingsOverlay') closeSettings();
                });

                // Click on message area to focus input
                messageArea.addEventListener('click', () => {
                    terminalInput.focus();
                });

                // Close palette when clicking outside
                document.addEventListener('click', (e) => {
                    if (!inputArea.contains(e.target) && !commandPalette.contains(e.target) && !messageArea
                        .contains(e.target)) {
                        closePalette();
                    }
                });
            }

            // ============ INPUT HANDLING ============
            function handleInput(e) {
                const value = terminalInput.value;
                const cursorPos = terminalInput.selectionStart || 0;

                // Check if we should open the palette
                const textBeforeCursor = value.substring(0, cursorPos);
                const startsWithSlash = textBeforeCursor.startsWith('/') &&
                    !textBeforeCursor.includes(' ') &&
                    !textBeforeCursor.includes('\n');

                if (startsWithSlash) {
                    const searchTerm = textBeforeCursor.substring(1).toLowerCase();
                    filterCommands(searchTerm);
                    openPalette();
                } else {
                    closePalette();
                }

                updatePrompt();
            }

            function handleKeydown(e) {
                // Close palette on Escape
                if (e.key === 'Escape') {
                    if (isPaletteOpen) {
                        closePalette();
                        e.preventDefault();
                    }
                    return;
                }

                // Palette navigation
                if (isPaletteOpen && filteredCommands.length > 0) {
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        paletteSelectedIndex = (paletteSelectedIndex + 1) % filteredCommands.length;
                        updatePaletteHighlight();
                        scrollPaletteToSelected();
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        paletteSelectedIndex = (paletteSelectedIndex - 1 + filteredCommands.length) %
                            filteredCommands.length;
                        updatePaletteHighlight();
                        scrollPaletteToSelected();
                        return;
                    }
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const cmd = filteredCommands[paletteSelectedIndex];
                        if (cmd) {
                            terminalInput.value = cmd.name + ' ';
                            closePalette();
                            updatePrompt();
                            terminalInput.focus();
                            terminalInput.setSelectionRange(terminalInput.value.length, terminalInput.value
                                .length);
                        }
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const cmd = filteredCommands[paletteSelectedIndex];
                        if (cmd) {
                            terminalInput.value = cmd.name + ' ';
                            closePalette();
                            updatePrompt();
                            // Don't submit, let user add args
                            terminalInput.focus();
                            terminalInput.setSelectionRange(terminalInput.value.length, terminalInput.value
                                .length);
                        }
                        return;
                    }
                }

                // Enter to submit
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (isPaletteOpen) {
                        // If palette open and enter pressed, select the command
                        const cmd = filteredCommands[paletteSelectedIndex];
                        if (cmd) {
                            terminalInput.value = cmd.name + ' ';
                            closePalette();
                            updatePrompt();
                            terminalInput.focus();
                            terminalInput.setSelectionRange(terminalInput.value.length, terminalInput.value
                                .length);
                        }
                    } else {
                        submitInput();
                    }
                    return;
                }

                // Arrow up/down for history (when palette is not open)
                if (!isPaletteOpen) {
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        if (historyIndex < commandHistory.length - 1) {
                            historyIndex++;
                            terminalInput.value = commandHistory[commandHistory.length - 1 - historyIndex] ||
                                '';
                        }
                        updatePrompt();
                        return;
                    }
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        if (historyIndex > 0) {
                            historyIndex--;
                            terminalInput.value = commandHistory[commandHistory.length - 1 - historyIndex] ||
                                '';
                        } else if (historyIndex === 0) {
                            historyIndex = -1;
                            terminalInput.value = '';
                        }
                        updatePrompt();
                        return;
                    }
                }

                // Tab for autocomplete even without palette
                if (e.key === 'Tab' && !isPaletteOpen) {
                    const value = terminalInput.value;
                    if (value.startsWith('/') && !value.includes(' ')) {
                        e.preventDefault();
                        const matches = COMMANDS.filter(c => c.name.startsWith(value));
                        if (matches.length === 1) {
                            terminalInput.value = matches[0].name + ' ';
                            updatePrompt();
                        } else if (matches.length > 1) {
                            filterCommands(value.substring(1));
                            openPalette();
                        }
                    }
                    return;
                }
            }

            // ============ SUBMIT ============
            function submitInput() {
                const value = terminalInput.value.trim();
                if (!value) return;

                commandHistory.push(value);
                historyIndex = -1;
                terminalInput.value = '';

                if (value.startsWith('/')) {
                    handleCommand(value);
                } else {
                    handleChatMessage(value);
                }
                updatePrompt();
            }

            // ============ CHAT MESSAGES ============
            function handleChatMessage(text) {
                addMessage('user', text, 'user');
                chatHistory.push({ role: 'user', content: text });
                scrollToBottom();
                touchCurrentChat();
                simulateTyping(() => apiPost('/agent/chat', { text, messages: chatHistory.slice(0, -1) }, { user_id: currentUser })
                    .then(body => {
                        const answer = (body.data && body.data.answer) || '(no answer)';
                        chatHistory.push({ role: 'assistant', content: answer });
                        addMessage('ai', answer, 'ai');
                        touchCurrentChat();
                    })
                    .catch(err => {
                        chatHistory.pop();
                        addMessage('error', err.message, 'error');
                    })
                    .finally(() => scrollToBottom()));
            }

            // ============ COMMAND HANDLING ============
            function handleCommand(input) {
                const parts = input.split(' ');
                const commandName = parts[0].toLowerCase();
                const args = parts.slice(1).join(' ');

                const command = COMMANDS.find(c => c.name === commandName);

                if (command) {
                    addMessage('user', input, 'user');
                    scrollToBottom();
                    setTimeout(() => {
                        command.action(args);
                        scrollToBottom();
                    }, 200);
                } else {
                    // Unknown command
                    addMessage('user', input, 'user');
                    addMessage('error', `Command not found: ${commandName}`, 'error');
                    addMessage('system', 'Type /help to see all available commands.', 'system');
                    scrollToBottom();
                }
            }

            // ============ COMMAND ACTIONS ============
            function showHelp() {
                const lines = [
                    { prompt: 'system', text: '━━━ Available Commands ━━━', type: 'system' },
                ];
                COMMANDS.forEach(cmd => {
                    lines.push({ prompt: 'cmd', text: `  ${cmd.icon}  ${cmd.name.padEnd(12)} — ${cmd.desc}`,
                        type: 'system' });
                });
                lines.push({ prompt: 'system', text: '━━━━━━━━━━━━━━━━━━━━━━━━━━━', type: 'system' });
                lines.push({ prompt: 'ai', text: 'You can also just type a regular message to chat with me! 💬',
                    type: 'ai' });
                lines.forEach((line, i) => {
                    setTimeout(() => {
                        addMessage(line.prompt, line.text, line.type);
                        scrollToBottom();
                    }, i * 40);
                });
            }

            function clearTerminal() {
                messageArea.innerHTML = '';
                messageCount = 0;
                addMessage('system', 'Terminal cleared. Fresh start! ✨', 'system');
                scrollToBottom();
            }

            function applyTheme(t, silent) {
                theme = t === 'light' ? 'light' : 'dark';
                terminalContainer.setAttribute('data-theme', theme);
                const input = document.getElementById('setTheme');
                if (input) input.value = theme;
                document.body.style.background = theme === 'light' ? '#d5dde5' : '#05070a';
                localStorage.setItem('nmd_theme', theme);
                if (!silent) {
                    addMessage('system', `Theme switched to ${theme} mode. ${theme === 'dark' ? '🌙' : '☀️'}`,
                        'system');
                    scrollToBottom();
                }
            }

            function toggleTheme() {
                applyTheme(theme === 'dark' ? 'light' : 'dark');
            }

            // ============ NEBULONMIND COMMANDS ============
            function validUsername(name) {
                name = (name || '').trim();
                return !!name && !name.startsWith('/');
            }

            function createUser(args) {
                const name = (args || '').trim();
                if (!validUsername(name)) {
                    addMessage('warning', 'Usage: /create <username> — e.g. /create sathya', 'warning');
                    return;
                }
                addMessage('user', `/create ${name}`, 'user');
                apiPost('/user/create_user', { username: name })
                    .then(body => {
                        const d = body.data || {};
                        currentUser = d.username || name;
                        localStorage.setItem('nmd_user', currentUser);
                        resetChatContext();
                        loadChatsFromServer();
                        addMessage('success',
                            `User \`${currentUser}\` ${d.created ? 'created' : 'already existed'} ` +
                            `(user_id ${d.user_id}). Now chatting as \`${currentUser}\`.`,
                            'success');
                    })
                    .catch(err => addMessage('error', `Create failed: ${err.message}`, 'error'));
            }

            function setupUser(args) {
                const name = (args || '').trim();
                if (!validUsername(name)) {
                    addMessage('warning', 'Usage: /setup <username> (must already be registered)', 'warning');
                    return;
                }
                addMessage('user', `/setup ${name}`, 'user');
                apiPost('/user/setup', { username: name })
                    .then(() => {
                        currentUser = name;
                        localStorage.setItem('nmd_user', currentUser);
                        resetChatContext();
                        loadChatsFromServer();
                        addMessage('success', `Now chatting as existing user \`${currentUser}\`.`, 'success');
                    })
                    .catch(err => addMessage('error',
                        `Setup failed: ${err.message} — use /create to register it.`, 'error'));
            }

            function whoami() {
                addMessage('ai', `Currently chatting as \`${currentUser}\`.`);
            }

            function showServerStatus() {
                apiGet('/health').then(body => {
                    const d = body.data || {};
                    addMessage('system', '━━━ Server Status ━━━', 'system');
                    addMessage('ai',
                        `Service:   ${d.service || 'NebulonMind'}\n` +
                        `Version:   ${d.version || '?'}\n` +
                        `Backend:   ${d.backend || '?'}\n` +
                        `LLM:       ${d.provider || '—'}\n` +
                        `Users:     ${d.minds !== undefined ? d.minds : '?'}\n` +
                        `Chatting:  ${currentUser}`);
                    addMessage('system', '━━━━━━━━━━━━━━━━━━━━━━━━━━', 'system');
                }).catch(err => addMessage('error', `Status failed: ${err.message}`, 'error'));
            }

            function searchMemories(args) {
                const query = (args || '').trim();
                if (!query) {
                    addMessage('warning', 'Usage: /search <natural-language query>', 'warning');
                    return;
                }
                addMessage('user', `/search ${query}`, 'user');
                apiGet('/search', { query, top_k: 5, user_id: currentUser }).then(body => {
                    const results = (body.data && body.data.results) || [];
                    if (!results.length) {
                        addMessage('ai', 'No memories recalled for that query.', 'ai');
                        return;
                    }
                    addMessage('system', `━━━ Recall: "${query}" (${results.length}) ━━━`, 'system');
                    results.forEach((m, i) => {
                        const cls = m.classification || {};
                        const imp = m.importance || {};
                        const score = imp.score !== undefined
                            ? `score=${imp.score} (${imp.priority || ''})`
                            : (imp.priority || '');
                        addMessage('ai',
                            `[${i + 1}] ${m.content.text || ''}\n` +
                            `    type=${cls.memory_type || cls.category || ''}  ${score}`,
                            'ai');
                    });
                }).catch(err => addMessage('error', `Search failed: ${err.message}`, 'error'));
            }

            function showContext(args) {
                const query = (args || '').trim();
                if (!query) {
                    addMessage('warning', 'Usage: /context <query>', 'warning');
                    return;
                }
                apiPost('/memory/context', {}, { query, top_k: 5, max_characters: 2000, user_id: currentUser })
                    .then(body => {
                        const d = body.data || {};
                        addMessage('system', `━━━ Context: "${query}" ━━━`, 'system');
                        addMessage('ai', d.context || '(empty context)', 'ai');
                    })
                    .catch(err => addMessage('error', `Context failed: ${err.message}`, 'error'));
            }

            function showSessions() {
                apiGet('/agent/sessions', { user_id: currentUser }).then(body => {
                    const sessions = (body.data && body.data.sessions) || [];
                    if (!sessions.length) {
                        addMessage('ai', 'No active agent sessions.', 'ai');
                        return;
                    }
                    addMessage('system', '━━━ Agent Sessions ━━━', 'system');
                    sessions.forEach(s =>
                        addMessage('system',
                            ` ${s.session_id}  •  ${s.message_count} msgs  •  ${s.status}`, 'system'));
                }).catch(err => addMessage('error', `Sessions failed: ${err.message}`, 'error'));
            }

            function showLLMStatus() {
                apiGet('/llm/status').then(body => {
                    const d = body.data || {};
                    addMessage('ai',
                        `LLM provider: ${d.provider || '—'}\n` +
                        `State:        ${d.configured ? 'configured' : 'not configured'}` +
                        (d.model ? `\nModel:        ${d.model}` : '') +
                        (d.error ? `\nError:        ${d.error}` : ''));
                }).catch(err => addMessage('error', `LLM status failed: ${err.message}`, 'error'));
            }

            function backgroundStatus() {
                apiGet('/background/status').then(body => {
                    const d = body.data || {};
                    const jobs = (d.jobs || []).map(j => `${j.job_id} (${j.runs} runs)`).join(', ') || '—';
                    addMessage('ai',
                        `Scheduler running: ${d.running}\n` +
                        `Poll every:  ${d.poll_seconds}s\n` +
                        `Jobs:        ${jobs}`);
                }).catch(err => addMessage('error', `Background status failed: ${err.message}`, 'error'));
            }

            function runEval() {
                addMessage('user', '/eval', 'user');
                addMessage('system', 'Running evaluation — this performs live LLM calls and may take a while.', 'system');
                simulateTyping(() => apiPost('/evaluation/run', { dataset: 'memory_test_v1', seed: true, max_items: 10 })
                    .then(body => {
                        const d = body.data || {};
                        const m = d.metrics || {};
                        addMessage('system', `━━━ Evaluation: ${d.dataset || 'memory_test_v1'} ━━━`, 'system');
                        addMessage('ai',
                            `Questions:      ${m.questions}\n` +
                            `Retrieval acc:  ${((m.retrieval_accuracy || 0) * 100).toFixed(0)}%\n` +
                            `Tool acc:       ${((m.tool_selection_accuracy || 0) * 100).toFixed(0)}%\n` +
                            `Answer acc:     ${((m.answer_correctness || 0) * 100).toFixed(0)}%\n` +
                            `Hallucination:  ${((m.hallucination_rate || 0) * 100).toFixed(1)}%\n` +
                            `Avg latency:    ${(m.avg_latency_ms || 0).toFixed(0)} ms`,
                            'ai');
                        addMessage('system', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'system');
                    })
                    .catch(err => addMessage('error', `Evaluation failed: ${err.message}`, 'error'))
                    .finally(() => scrollToBottom()));
            }

            // ============ CHAT HISTORY ============
            function renderChatHistory() {
                const holder = document.getElementById('chatHistoryList');
                if (!holder) return;
                const mine = chatsForUser(currentUser);
                if (!mine.length) {
                    holder.innerHTML = '<div class="settings-note">No saved chats for this user yet.</div>';
                    return;
                }
                holder.innerHTML = mine.map(c => `
                    <div class="chat-history-item">
                        <button class="chat-history-restore" data-id="${c.id}">${escHtml(c.title)}</button>
                        <button class="chat-history-del" data-id="${c.id}" title="Delete chat">✕</button>
                    </div>`).join('');
                holder.querySelectorAll('.chat-history-restore').forEach(b =>
                    b.addEventListener('click', () => restoreChat(b.dataset.id)));
                holder.querySelectorAll('.chat-history-del').forEach(b =>
                    b.addEventListener('click', () => deleteChat(b.dataset.id)));
            }

            // ============ SETTINGS PANEL ============
            function escHtml(s) {
                return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
                    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
            }

            function openSettings() {
                document.getElementById('settingsOverlay').classList.add('visible');
                renderSettings();
            }

            function closeSettings() {
                document.getElementById('settingsOverlay').classList.remove('visible');
            }

            function renderSettings() {
                const body = document.getElementById('settingsBody');
                body.innerHTML = '<div class="settings-note">Loading settings…</div>';
                Promise.all([
                    apiGet('/config/cfg'),
                    apiGet('/llm/status').catch(() => ({ data: {} })),
                ]).then(([cfgRes, llmRes]) => {
                    const groups = (cfgRes.data && cfgRes.data.groups) || [];
                    const llm = llmRes.data || {};
                    let html = '';
                    html += `<div class="settings-group">
                        <div class="settings-group-title">Session</div>
                        <div class="settings-row">
                            <span class="settings-label">User</span>
                            <span class="settings-value">${escHtml(currentUser)}</span>
                        </div>
                        <div class="settings-row">
                            <span class="settings-label">Theme</span>
                            <select id="setTheme" class="settings-input">
                                <option value="dark">Dark</option>
                                <option value="light">Light</option>
                            </select>
                        </div>
                    </div>`;
                    html += `<div class="settings-group">
                        <div class="settings-group-title">LLM credentials</div>
                        <div class="settings-note">Stored in .env, never in cfg. Leave the API key blank to keep the current key.</div>
                        <div class="settings-row">
                            <span class="settings-label">Provider</span>
                            <input type="text" class="settings-input" data-cred="provider" value="${escHtml(llm.provider || '')}" placeholder="nvidia / ollama / openai / ...">
                        </div>
                        <div class="settings-row">
                            <span class="settings-label">Model</span>
                            <input type="text" class="settings-input" data-cred="model" value="${escHtml(llm.model || '')}" placeholder="model name">
                        </div>
                        <div class="settings-row">
                            <span class="settings-label">API key</span>
                            <input type="password" class="settings-input" data-cred="api_key" autocomplete="off" placeholder="${llm.configured ? '••••••  (key set — leave blank)' : 'not set'}">
                        </div>
                        <div class="settings-row">
                            <span class="settings-label">Base URL</span>
                            <input type="text" class="settings-input" data-cred="base_url" placeholder="https://… (optional)">
                        </div>
                    </div>`;
                    html += `<div class="settings-group">
                        <div class="settings-group-title">Chat history</div>
                        <div id="chatHistoryList" class="chat-history-list">…</div>
                        <button class="settings-btn settings-btn-small" id="btnNewChatInSettings">＋ New chat</button>
                    </div>`;
                    groups.forEach(g => {
                        if (g.id === 'background') return;
                        html += `<div class="settings-group">
                            <div class="settings-group-title">${escHtml(g.title)}</div>
                            <div class="settings-note">${escHtml(g.description || '')}</div>`;
                        g.keys.forEach(k => {
                            html += settingsRow(g.id, k);
                        });
                        html += `</div>`;
                    });
                    body.innerHTML = html;

                    const themeSelect = document.getElementById('setTheme');
                    if (themeSelect) {
                        themeSelect.value = theme;
                        themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));
                    }
                    const newChatBtn = document.getElementById('btnNewChatInSettings');
                    if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);
                    renderChatHistory();
                }).catch(err => {
                    body.innerHTML = `<div class="settings-note">Could not load settings: ${escHtml(err.message)}</div>`;
                });
            }

            function settingsRow(section, k) {
                const val = escHtml(String(k.value == null ? '' : k.value));
                const id = `set-${k.key}`;
                if (k.type === 'bool') {
                    const checked = k.value ? 'checked' : '';
                    return `<div class="settings-row">
                        <span class="settings-label">${escHtml(k.label)}</span>
                        <input type="checkbox" class="settings-check" id="${id}"
                            data-section="${escHtml(section)}" data-key="${escHtml(k.key)}" ${checked}>
                    </div>`;
                }
                const inputType = k.type === 'int' || k.type === 'float' ? 'number' : 'text';
                return `<div class="settings-row">
                    <span class="settings-label" title="${escHtml(k.key)}">${escHtml(k.label)}</span>
                    <input type="${inputType}" class="settings-input" id="${id}"
                        data-section="${escHtml(section)}" data-key="${escHtml(k.key)}" value="${val}">
                </div>`;
            }

            function settingsToConfig() {
                const config = {};
                document.querySelectorAll('#settingsBody [data-section][data-key]').forEach(el => {
                    const section = el.dataset.section;
                    const key = el.dataset.key;
                    if (!config[section]) config[section] = {};
                    if (el.type === 'checkbox') config[section][key] = el.checked ? 'true' : 'false';
                    else config[section][key] = el.value.trim();
                });
                return config;
            }

            function saveSettings() {
                const cfgConfig = settingsToConfig();
                const credPart = {};
                document.querySelectorAll('#settingsBody [data-cred]').forEach(el => {
                    const key = el.dataset.cred;
                    const value = el.type === 'checkbox' ? (el.checked ? 'true' : 'false') : el.value.trim();
                    if (key === 'api_key' && !value) return;
                    credPart[key] = value;
                });
                const tasks = [];
                if (Object.keys(cfgConfig).length) {
                    tasks.push(apiRequest('PUT', '/config/cfg', { body: { config: cfgConfig } }));
                }
                if (Object.keys(credPart).length) {
                    tasks.push(apiRequest('PUT', '/dashboard/config', { body: { config: { llm: credPart } } }));
                }
                if (!tasks.length) {
                    addMessage('system', 'No settings changed.', 'system');
                    closeSettings();
                    return;
                }
                Promise.all(tasks)
                    .then(() => {
                        addMessage('success',
                            'Settings saved and applied live. Use “Restart service” for provider/bind changes.',
                            'success');
                        scrollToBottom();
                        closeSettings();
                        refreshApiBadge();
                    })
                    .catch(err => addMessage('error', `Save failed: ${err.message}`, 'error'));
            }

            function restartService() {
                apiPost('/config/restart')
                    .then(() => {
                        closeSettings();
                        addMessage('system', 'Restarting service… the console will reconnect shortly.', 'system');
                        setTimeout(() => window.location.reload(), 11000);
                    })
                    .catch(err => addMessage('error', `Restart failed: ${err.message}`, 'error'));
            }

            // ============ PALETTE ============
            function openPalette() {
                if (isPaletteOpen) return;
                isPaletteOpen = true;
                paletteSelectedIndex = 0;
                commandPalette.classList.add('visible');
                renderPalette();
                updatePaletteHighlight();
            }

            function closePalette() {
                if (!isPaletteOpen) return;
                isPaletteOpen = false;
                commandPalette.classList.remove('visible');
            }

            function filterCommands(searchTerm) {
                if (!searchTerm) {
                    filteredCommands = [...COMMANDS];
                } else {
                    filteredCommands = COMMANDS.filter(c =>
                        c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                        c.desc.toLowerCase().includes(searchTerm.toLowerCase())
                    );
                }
                paletteSelectedIndex = 0;
                renderPalette();
                updatePaletteHighlight();
            }

            function renderPalette() {
                if (filteredCommands.length === 0) {
                    paletteItems.innerHTML = `
                <div class="palette-item" style="cursor: default; opacity: 0.6;">
                  <div class="palette-info">
                    <div class="palette-name">No commands found</div>
                  </div>
                </div>
              `;
                    return;
                }
                paletteItems.innerHTML = filteredCommands.map((cmd, index) => `
              <div class="palette-item" data-index="${index}">
                <div class="palette-icon">${cmd.icon}</div>
                <div class="palette-info">
                  <div class="palette-name">${cmd.name}</div>
                  <div class="palette-desc">${cmd.desc}</div>
                </div>
                <div class="palette-shortcut">${cmd.shortcut}</div>
              </div>
            `).join('');
            }

            function updatePaletteHighlight() {
                const items = paletteItems.querySelectorAll('.palette-item');
                items.forEach((item, index) => {
                    if (index === paletteSelectedIndex) {
                        item.classList.add('selected');
                    } else {
                        item.classList.remove('selected');
                    }
                });
            }

            function scrollPaletteToSelected() {
                const items = paletteItems.querySelectorAll('.palette-item');
                const selected = items[paletteSelectedIndex];
                if (selected) {
                    selected.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                }
            }

            function selectPaletteCommand(index) {
                const cmd = filteredCommands[index];
                if (cmd) {
                    terminalInput.value = cmd.name + ' ';
                    closePalette();
                    updatePrompt();
                    terminalInput.focus();
                    terminalInput.setSelectionRange(terminalInput.value.length, terminalInput.value.length);
                }
            }

            // ============ MESSAGE RENDERING ============
            function addMessage(prompt, text, type = 'system') {
                messageCount++;
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';

                const lineDiv = document.createElement('div');
                lineDiv.className = 'message-line';

                const promptSpan = document.createElement('span');
                promptSpan.className = `message-prompt ${prompt}-prompt`;
                const promptSymbols = {
                    'user': '❯',
                    'ai': '◉',
                    'system': '◆',
                    'error': '✖',
                    'success': '✔',
                    'warning': '⚠',
                    'cmd': '›',
                };
                promptSpan.textContent = promptSymbols[prompt] || '◆';

                const textSpan = document.createElement('span');
                textSpan.className = `message-text ${type}-text`;
                textSpan.textContent = text;

                lineDiv.appendChild(promptSpan);
                lineDiv.appendChild(textSpan);
                messageDiv.appendChild(lineDiv);
                messageArea.appendChild(messageDiv);

                // Auto scroll if near bottom
                const nearBottom = messageArea.scrollHeight - messageArea.scrollTop - messageArea.clientHeight < 100;
                if (nearBottom) {
                    scrollToBottom();
                }
            }

            function scrollToBottom() {
                messageArea.scrollTop = messageArea.scrollHeight;
            }

            function simulateTyping(callback, delay = 800) {
                if (document.querySelector('.typing-indicator')) return;
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message';
                const lineDiv = document.createElement('div');
                lineDiv.className = 'message-line';
                const promptSpan = document.createElement('span');
                promptSpan.className = 'message-prompt ai-prompt';
                promptSpan.textContent = '◉';
                const typingSpan = document.createElement('span');
                typingSpan.className = 'typing-indicator';
                typingSpan.innerHTML = '<span></span><span></span><span></span>';
                lineDiv.appendChild(promptSpan);
                lineDiv.appendChild(typingSpan);
                typingDiv.appendChild(lineDiv);
                messageArea.appendChild(typingDiv);
                scrollToBottom();

                const stopTyping = () => {
                    typingDiv.remove();
                    scrollToBottom();
                };
                const randomDelay = delay + Math.random() * 500;
                setTimeout(() => {
                    try {
                        const result = callback();
                        if (result && typeof result.then === 'function') {
                            result.then(stopTyping, stopTyping);
                        } else {
                            stopTyping();
                        }
                    } catch (e) {
                        stopTyping();
                    }
                }, randomDelay);
            }

            // ============ PROMPT UPDATE ============
            function updatePrompt() {
                const value = terminalInput.value;
                const cursorPos = terminalInput.selectionStart || 0;
                const textBeforeCursor = value.substring(0, cursorPos);
                const startsWithSlash = textBeforeCursor.startsWith('/') &&
                    !textBeforeCursor.includes(' ') &&
                    !textBeforeCursor.includes('\n');

                if (startsWithSlash) {
                    inputPrompt.style.color = 'var(--accent)';
                    inputPrompt.textContent = '❯';
                } else {
                    inputPrompt.style.color = 'var(--prompt-color)';
                    inputPrompt.textContent = '❯';
                }
            }

            // ============ GLOBAL KEYBOARD SHORTCUT ============
            document.addEventListener('keydown', (e) => {
                // Ctrl+L to clear
                if (e.ctrlKey && e.key === 'l') {
                    e.preventDefault();
                    clearTerminal();
                }
                // Ctrl+K to focus input
                if (e.ctrlKey && e.key === 'k') {
                    e.preventDefault();
                    terminalInput.focus();
                }
                // Auto-focus "/" typing
                if (e.key === '/' && document.activeElement !== terminalInput) {
                    e.preventDefault();
                    terminalInput.value = '/';
                    terminalInput.focus();
                    handleInput({ target: terminalInput });
                }
            });

            // ============ START ============
            init();
        })();
