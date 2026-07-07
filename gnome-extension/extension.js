import GLib from 'gi://GLib'
import Gio from 'gi://Gio'
import GObject from 'gi://GObject'
import St from 'gi://St'
import Clutter from 'gi://Clutter'
import Shell from 'gi://Shell'

import { Extension, gettext as _ } from 'resource:///org/gnome/shell/extensions/extension.js'
import * as Main from 'resource:///org/gnome/shell/ui/main.js'
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js'
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js'

import {
  WhisperClient,
  HuggingFaceClient,
  listLocalModels,
  listLocalLlmModels,
  downloadModel,
  downloadLlmModel,
  writeServiceOverride,
  writePttServiceOverride,
  restartService,
  typeText,
  backspaceN,
  saveToFile,
  logDebug
} from './utils.js'

const DEVICES = ['NPU', 'CPU', 'GPU']
const BACKENDS = ['openvino', 'whisper-cpp']
const HOTKEYS = [
  'KEY_RIGHTCTRL',
  'KEY_RIGHTALT',
  'KEY_RIGHTSHIFT',
  'KEY_SCROLLLOCK',
  'KEY_PAUSE'
]
const DEFAULT_TONES = ['diplomatic', 'professional']
const LANGUAGES = [
  ['Auto', ''],
  ['English', 'en'],
  ['German', 'de'],
  ['French', 'fr'],
  ['Spanish', 'es'],
  ['Italian', 'it'],
  ['Portuguese', 'pt'],
  ['Dutch', 'nl'],
  ['Polish', 'pl'],
  ['Japanese', 'ja'],
  ['Chinese', 'zh'],
  ['Korean', 'ko']
]
const TRANSLATE_TARGETS = [
  ['Disabled', ''],
  ['English', 'English'],
  ['German', 'German'],
  ['French', 'French'],
  ['Spanish', 'Spanish'],
  ['Italian', 'Italian'],
  ['Portuguese', 'Portuguese'],
  ['Dutch', 'Dutch'],
  ['Polish', 'Polish'],
  ['Japanese', 'Japanese'],
  ['Chinese', 'Chinese'],
  ['Korean', 'Korean']
]

const DBUS_IFACE = `
<node>
  <interface name="com.whisper.LanguageBuddy">
    <method name="HandleTranscription">
      <arg type="s" direction="in" name="text"/>
      <arg type="b" direction="out" name="handled"/>
    </method>
    <method name="HandleTranscriptionWithContext">
      <arg type="s" direction="in" name="text"/>
      <arg type="s" direction="in" name="context_json"/>
      <arg type="b" direction="out" name="handled"/>
    </method>
    <method name="GetFocusedApp">
      <arg type="s" direction="out" name="app_id"/>
      <arg type="s" direction="out" name="window_title"/>
    </method>
    <method name="ShowHistoryPicker">
      <arg type="s" direction="in" name="items_json"/>
      <arg type="b" direction="out" name="shown"/>
    </method>
  </interface>
</node>`

const OVERLAY_PADDING = 100

// ---------------------------------------------------------------------------
// Language Buddy overlay — floating card list at bottom-right
// ---------------------------------------------------------------------------

const LanguageBuddyOverlay = GObject.registerClass(
  class LanguageBuddyOverlay extends St.BoxLayout {
    _init () {
      super._init({
        vertical: true,
        style_class: 'whisper-overlay',
        reactive: true,
        visible: false
      })

      this._timeoutId = null
    }

    show (tones, originalText, onSelect, timeoutSec) {
      this.destroy_all_children()
      this._cards = {}
      this._onSelect = onSelect

      const headerBox = new St.BoxLayout({ x_expand: true })
      const headerLabel = new St.Label({
        text: _('Language Buddy'),
        style_class: 'whisper-overlay-header',
        x_expand: true
      })
      const dismissBtn = new St.Button({
        style_class: 'whisper-overlay-dismiss',
        child: new St.Label({ text: '✕' })
      })
      dismissBtn.connect('clicked', () => this._dismiss())
      headerBox.add_child(headerLabel)
      headerBox.add_child(dismissBtn)
      this.add_child(headerBox)

      this._addCard('original', originalText, true)
      for (const tone of tones) {
        this._addCard(tone, _('Processing...'), false)
      }

      this._showAndPosition()

      if (timeoutSec > 0) {
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutSec, () => {
          this._dismiss()
          return GLib.SOURCE_REMOVE
        })
      }
    }

    _addCard (tone, text, ready) {
      const card = new St.Button({
        style_class: 'whisper-overlay-card',
        x_expand: true,
        reactive: ready,
        can_focus: false,
        track_hover: true
      })

      const cardBox = new St.BoxLayout({ vertical: true })

      const toneLabel = new St.Label({
        text: tone.charAt(0).toUpperCase() + tone.slice(1),
        style_class: 'whisper-overlay-tone'
      })
      cardBox.add_child(toneLabel)

      const textLabel = new St.Label({
        text,
        style_class: 'whisper-overlay-text'
      })
      textLabel.clutter_text.set_line_wrap(true)
      textLabel.clutter_text.set_ellipsize(0)
      cardBox.add_child(textLabel)

      card.set_child(cardBox)

      const entry = { card, textLabel, text }
      this._cards[tone] = entry

      card.connect('clicked', () => {
        if (this._onSelect && entry.card.reactive) {
          const selectedText = entry.text
          const callback = this._onSelect
          this._dismiss()
          callback(selectedText)
        }
      })

      this.add_child(card)
    }

    updateCard (tone, text) {
      const entry = this._cards[tone]
      if (!entry) return
      entry.text = text
      entry.textLabel.set_text(text)
      entry.card.reactive = true
      this._reposition()
    }

    _showAndPosition () {
      const monitor = Main.layoutManager.primaryMonitor
      if (!monitor) return

      const maxH = monitor.height - OVERLAY_PADDING * 2
      this.style = `max-height: ${maxH}px;`

      this.set_position(monitor.x + monitor.width, monitor.y + monitor.height)
      this.visible = true

      this._reposition()
    }

    _reposition () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      this._positionId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 50, () => {
        this._positionId = null
        const monitor = Main.layoutManager.primaryMonitor
        if (!monitor) return GLib.SOURCE_REMOVE
        this.set_position(
          monitor.x + monitor.width - this.width - OVERLAY_PADDING,
          monitor.y + monitor.height - this.height - OVERLAY_PADDING
        )
        return GLib.SOURCE_REMOVE
      })
    }

    _dismiss () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      if (this._timeoutId) {
        GLib.source_remove(this._timeoutId)
        this._timeoutId = null
      }
      this.visible = false
      this.style = null
      this._cards = {}
      this._onSelect = null
      this.destroy_all_children()
    }

    destroy () {
      this._dismiss()
      super.destroy()
    }
  }
)

// ---------------------------------------------------------------------------
// History picker overlay — floating card list at bottom-right
// ---------------------------------------------------------------------------

const HistoryOverlay = GObject.registerClass(
  class HistoryOverlay extends St.BoxLayout {
    _init () {
      super._init({
        vertical: true,
        style_class: 'whisper-overlay',
        reactive: true,
        visible: false
      })
      this._timeoutId = null
    }

    show (items, timeoutSec = 10) {
      this.destroy_all_children()

      const headerBox = new St.BoxLayout({ x_expand: true })
      const headerLabel = new St.Label({
        text: _('Transcription History'),
        style_class: 'whisper-overlay-header',
        x_expand: true
      })
      const dismissBtn = new St.Button({
        style_class: 'whisper-overlay-dismiss',
        child: new St.Label({ text: '✕' })
      })
      dismissBtn.connect('clicked', () => this._dismiss())
      headerBox.add_child(headerLabel)
      headerBox.add_child(dismissBtn)
      this.add_child(headerBox)

      for (const item of items) {
        const card = new St.Button({
          style_class: 'whisper-overlay-card',
          x_expand: true,
          reactive: true,
          track_hover: true
        })

        const cardBox = new St.BoxLayout({ vertical: true })

        const ago = this._timeAgo(item.ts)
        const timeLabel = new St.Label({
          text: ago,
          style_class: 'whisper-overlay-tone'
        })
        cardBox.add_child(timeLabel)

        const textLabel = new St.Label({
          text: item.text.length > 120 ? item.text.substring(0, 120) + '...' : item.text,
          style_class: 'whisper-overlay-text'
        })
        textLabel.clutter_text.set_line_wrap(true)
        textLabel.clutter_text.set_ellipsize(0)
        cardBox.add_child(textLabel)

        card.set_child(cardBox)
        const fullText = item.text
        card.connect('clicked', () => {
          this._dismiss()
          typeText(fullText).catch(e => logDebug(`typeText failed: ${e.message}`))
        })
        this.add_child(card)
      }

      this._showAndPosition()

      if (timeoutSec > 0) {
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutSec, () => {
          this._dismiss()
          return GLib.SOURCE_REMOVE
        })
      }
    }

    _timeAgo (ts) {
      const now = Date.now() / 1000
      const diff = now - ts
      if (diff < 60) return 'just now'
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
      return `${Math.floor(diff / 86400)}d ago`
    }

    _showAndPosition () {
      const monitor = Main.layoutManager.primaryMonitor
      if (!monitor) return
      const maxH = monitor.height - OVERLAY_PADDING * 2
      this.style = `max-height: ${maxH}px;`
      this.set_position(monitor.x + monitor.width, monitor.y + monitor.height)
      this.visible = true
      this._reposition()
    }

    _reposition () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      this._positionId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 50, () => {
        this._positionId = null
        const monitor = Main.layoutManager.primaryMonitor
        if (!monitor) return GLib.SOURCE_REMOVE
        this.set_position(
          monitor.x + monitor.width - this.width - OVERLAY_PADDING,
          monitor.y + monitor.height - this.height - OVERLAY_PADDING
        )
        return GLib.SOURCE_REMOVE
      })
    }

    _dismiss () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      if (this._timeoutId) {
        GLib.source_remove(this._timeoutId)
        this._timeoutId = null
      }
      this.visible = false
      this.style = null
      this.destroy_all_children()
    }

    destroy () {
      this._dismiss()
      super.destroy()
    }
  }
)

// ---------------------------------------------------------------------------
// Panel indicator with dropdown menu
// ---------------------------------------------------------------------------

const WhisperIndicator = GObject.registerClass(
  class WhisperIndicator extends PanelMenu.Button {
    _init (extension) {
      super._init(0.0, _('Whisper NPU'))

      this._extension = extension
      this._settings = extension.getSettings()
      this._connected = false
      this._currentModel = null
      this._cachedRemoteModels = []
      this._cachedRemoteLlmModels = []
      this._cacheReady = false
      this._currentLlmModel = null

      this._client = new WhisperClient(
        this._settings.get_string('server-host'),
        this._settings.get_int('server-port')
      )
      this._hfClient = new HuggingFaceClient()

      const iconPath = GLib.build_filenamev([extension.path, 'icons', 'whisper-npu-symbolic.svg'])
      const gicon = Gio.icon_new_for_string(iconPath)
      const icon = new St.Icon({
        gicon,
        style_class: 'system-status-icon'
      })
      this.add_child(icon)

      this._buildMenu()

      this.menu.connect('open-state-changed', (_menu, isOpen) => {
        if (isOpen) this._refreshStatus()
      })

      this._settingsChangedId = this._settings.connect('changed', (settings, key) => {
        if (key === 'server-host' || key === 'server-port') {
          this._client.destroy()
          this._client = new WhisperClient(
            settings.get_string('server-host'),
            settings.get_int('server-port')
          )
        }
      })

      // Overlays
      this._overlay = new LanguageBuddyOverlay()
      Main.layoutManager.addChrome(this._overlay, {
        affectsStruts: false,
        trackFullscreen: false
      })

      this._historyOverlay = new HistoryOverlay()
      Main.layoutManager.addChrome(this._historyOverlay, {
        affectsStruts: false,
        trackFullscreen: false
      })

      // D-Bus
      this._dbusImpl = Gio.DBusExportedObject.wrapJSObject(DBUS_IFACE, this)
      this._dbusImpl.export(Gio.DBus.session, '/com/whisper/LanguageBuddy')

      this._preloadCache()
    }

    async _preloadCache () {
      try {
        const org = this._settings.get_string('hf-org')
        const [health, remoteModels, remoteLlmModels, llmInfo] = await Promise.all([
          this._client.getHealth(),
          this._hfClient.searchModels(org),
          this._hfClient.searchLlmModels(org),
          this._client.getLlmModels()
        ])

        if (health) {
          this._connected = true
          this._currentModel = health.model
        }

        this._currentLlmModel = llmInfo ? llmInfo.current : null

        this._cachedRemoteModels = remoteModels || []
        this._cachedRemoteLlmModels = remoteLlmModels || []
        this._cacheReady = true
        logDebug(`Preloaded ${this._cachedRemoteModels.length} STT and ${this._cachedRemoteLlmModels.length} LLM remote models`)

        this._populateModelSection()
        this._populateLlmModelSection()
      } catch (e) {
        logDebug(`Preload failed: ${e.message}`)
        this._cacheReady = true
      }
    }

    // -- D-Bus method handler -----------------------------------------------

    HandleTranscription (text) {
      const enabled = this._settings.get_boolean('language-buddy-enabled')
      if (!enabled) return false

      logDebug(`Language Buddy received: ${text.substring(0, 80)}`)
      this._processTranscription(text)
      return true
    }

    HandleTranscriptionWithContext (text, contextJson) {
      const enabled = this._settings.get_boolean('language-buddy-enabled')
      if (!enabled) return false

      logDebug(`Context-aware transcription: ${text.substring(0, 80)}`)
      try {
        const context = JSON.parse(contextJson)
        const tones = context.tones && context.tones.length > 0 ? context.tones : DEFAULT_TONES
        this._processTranscription(text, tones)
      } catch (e) {
        logDebug(`Context parse error: ${e.message}`)
        this._processTranscription(text)
      }
      return true
    }

    GetFocusedApp () {
      try {
        const focusWindow = global.display.focus_window
        if (!focusWindow) return ['', '']
        const tracker = Shell.WindowTracker.get_default()
        const app = tracker.get_window_app(focusWindow)
        const appId = app ? app.get_id() : ''
        const title = focusWindow.get_title() || ''
        return [appId, title]
      } catch (e) {
        logDebug(`GetFocusedApp error: ${e.message}`)
        return ['', '']
      }
    }

    ShowHistoryPicker (itemsJson) {
      try {
        const items = JSON.parse(itemsJson)
        if (!items || items.length === 0) return false
        this._historyOverlay.show(items, 10)
        return true
      } catch (e) {
        logDebug(`ShowHistoryPicker error: ${e.message}`)
        return false
      }
    }

    async _processTranscription (text, tones = null) {
      const bypass = this._settings.get_boolean('language-buddy-bypass')

      if (bypass) {
        typeText(text).catch(e => logDebug(`typeText failed: ${e.message}`))
      }

      const timeoutSec = this._settings.get_int('language-buddy-timeout')
      tones = tones || (text.length > 50
        ? [...DEFAULT_TONES, 'summarize']
        : DEFAULT_TONES)

      const onSelect = bypass
        ? async (selectedText) => {
            if (selectedText === text) return
            try {
              await backspaceN(text.length)
              await typeText(selectedText)
            } catch (e) {
              logDebug(`Replace failed: ${e.message}`)
            }
          }
        : async (selectedText) => {
            try {
              await typeText(selectedText)
            } catch (e) {
              logDebug(`typeText failed: ${e.message}`)
            }
          }

      this._overlay.show(tones, text, onSelect, timeoutSec)

      for (const tone of tones) {
        try {
          const result = await this._client.rewrite(text, [tone])
          if (result && result.variants) {
            const variant = result.variants.find(v => v.tone === tone)
            if (variant && !variant.error) {
              this._overlay.updateCard(tone, variant.text)
            } else {
              this._overlay.updateCard(tone, text)
            }
          } else {
            this._overlay.updateCard(tone, text)
          }
        } catch (e) {
          logDebug(`Rewrite (${tone}) failed: ${e.message}`)
          this._overlay.updateCard(tone, text)
        }
      }
    }

    // -- Menu ---------------------------------------------------------------

    _buildMenu () {
      this._statusItem = new PopupMenu.PopupMenuItem(_('Status: Checking...'), { reactive: false })
      this._statusItem.label.add_style_class_name('whisper-status-label')
      this.menu.addMenuItem(this._statusItem)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Language Buddy toggle
      this._buddyToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Language Buddy'),
        this._settings.get_boolean('language-buddy-enabled')
      )
      this._buddyToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('language-buddy-enabled', state)
        logDebug(`Language Buddy ${state ? 'enabled' : 'disabled'}`)
      })
      this.menu.addMenuItem(this._buddyToggle)

      this._bypassToggle = new PopupMenu.PopupSwitchMenuItem(
        _('  Bypass (type original, show suggestions)'),
        this._settings.get_boolean('language-buddy-bypass')
      )
      this._bypassToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('language-buddy-bypass', state)
        logDebug(`Language Buddy bypass ${state ? 'enabled' : 'disabled'}`)
      })
      this.menu.addMenuItem(this._bypassToggle)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // STT models (installed + available downloads)
      this._modelSection = new PopupMenu.PopupSubMenuMenuItem(_('Speech-to-Text Models'))
      this._modelSection.menu.addMenuItem(
        new PopupMenu.PopupMenuItem(_('Loading...'), { reactive: false })
      )
      this.menu.addMenuItem(this._modelSection)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // LLM model section
      this._llmModelSection = new PopupMenu.PopupSubMenuMenuItem(_('Language Buddy Models'))
      this._llmModelSection.menu.addMenuItem(
        new PopupMenu.PopupMenuItem(_('Loading...'), { reactive: false })
      )
      this.menu.addMenuItem(this._llmModelSection)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Server options
      this._deviceSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Device: ${this._settings.get_string('device')}`)
      )
      this._buildRadioGroup(this._deviceSection, DEVICES, 'device')
      this.menu.addMenuItem(this._deviceSection)

      this._backendSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Backend: ${this._settings.get_string('backend')}`)
      )
      this._buildRadioGroup(this._backendSection, BACKENDS, 'backend')
      this.menu.addMenuItem(this._backendSection)

      this._hotkeySection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Hotkey: ${this._formatHotkey(this._settings.get_string('hotkey'))}`)
      )
      this._buildHotkeyGroup(this._hotkeySection)
      this.menu.addMenuItem(this._hotkeySection)

      // Language selector
      const currentLang = this._settings.get_string('language')
      const langLabel = LANGUAGES.find(l => l[1] === currentLang)
      this._languageSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Language: ${langLabel ? langLabel[0] : 'Auto'}`)
      )
      this._buildLanguageGroup(this._languageSection)
      this.menu.addMenuItem(this._languageSection)

      // Recall key selector
      this._recallKeySection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Recall Key: ${this._formatHotkey(this._settings.get_string('recall-key'))}`)
      )
      this._buildRecallKeyGroup(this._recallKeySection)
      this.menu.addMenuItem(this._recallKeySection)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Feature toggles
      this._voiceCommandsToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Voice Commands'),
        this._settings.get_boolean('voice-commands-enabled')
      )
      this._voiceCommandsToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('voice-commands-enabled', state)
      })
      this.menu.addMenuItem(this._voiceCommandsToggle)

      this._notificationsToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Notifications'),
        this._settings.get_boolean('notifications-enabled')
      )
      this._notificationsToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('notifications-enabled', state)
      })
      this.menu.addMenuItem(this._notificationsToggle)

      this._autoPunctuateToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Auto-Punctuate'),
        this._settings.get_boolean('auto-punctuate')
      )
      this._autoPunctuateToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('auto-punctuate', state)
      })
      this.menu.addMenuItem(this._autoPunctuateToggle)

      this._audioFeedbackToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Audio Feedback'),
        this._settings.get_boolean('audio-feedback-enabled')
      )
      this._audioFeedbackToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('audio-feedback-enabled', state)
      })
      this.menu.addMenuItem(this._audioFeedbackToggle)

      this._formattingToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Dictation Formatting'),
        this._settings.get_boolean('dictation-formatting-enabled')
      )
      this._formattingToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('dictation-formatting-enabled', state)
      })
      this.menu.addMenuItem(this._formattingToggle)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Translate To submenu
      const currentTranslate = this._settings.get_string('translate-to')
      const translateLabel = TRANSLATE_TARGETS.find(t => t[1] === currentTranslate)
      this._translateSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Translate To: ${translateLabel ? translateLabel[0] : 'Disabled'}`)
      )
      this._buildTranslateGroup(this._translateSection)
      this.menu.addMenuItem(this._translateSection)

      // VAD Threshold submenu
      this._vadSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`VAD Threshold: ${this._settings.get_int('vad-threshold')} dB`)
      )
      const vadOptions = [
        ['-20 dB (aggressive)', -20],
        ['-30 dB', -30],
        ['-40 dB (default)', -40],
        ['-50 dB', -50],
        ['-60 dB (sensitive)', -60]
      ]
      const currentVad = this._settings.get_int('vad-threshold')
      for (const [label, value] of vadOptions) {
        const item = new PopupMenu.PopupMenuItem(label)
        if (value === currentVad) item.setOrnament(PopupMenu.Ornament.DOT)
        item.connect('activate', () => {
          this._settings.set_int('vad-threshold', value)
          this._vadSection.label.set_text(`VAD Threshold: ${value} dB`)
          const items = this._vadSection.menu._getMenuItems()
          for (const [i, mi] of items.entries()) {
            if (mi.label && i < vadOptions.length) {
              mi.setOrnament(vadOptions[i][1] === value ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE)
            }
          }
        })
        this._vadSection.menu.addMenuItem(item)
      }
      this.menu.addMenuItem(this._vadSection)

      // Stream Interval submenu
      this._streamIntervalSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Stream Interval: ${this._settings.get_double('stream-interval')}s`)
      )
      const intervalOptions = [
        ['1.0s (fast)', 1.0],
        ['2.0s', 2.0],
        ['3.0s (default)', 3.0],
        ['5.0s', 5.0],
        ['10.0s (slow)', 10.0]
      ]
      const currentInterval = this._settings.get_double('stream-interval')
      for (const [label, value] of intervalOptions) {
        const item = new PopupMenu.PopupMenuItem(label)
        if (value === currentInterval) item.setOrnament(PopupMenu.Ornament.DOT)
        item.connect('activate', () => {
          this._settings.set_double('stream-interval', value)
          this._streamIntervalSection.label.set_text(`Stream Interval: ${value}s`)
          const items = this._streamIntervalSection.menu._getMenuItems()
          for (const [i, mi] of items.entries()) {
            if (mi.label && i < intervalOptions.length) {
              mi.setOrnament(intervalOptions[i][1] === value ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE)
            }
          }
        })
        this._streamIntervalSection.menu.addMenuItem(item)
      }
      this.menu.addMenuItem(this._streamIntervalSection)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Export History submenu
      this._exportSection = new PopupMenu.PopupSubMenuMenuItem(_('Export History'))
      for (const [label, fmt] of [['JSON', 'json'], ['Markdown', 'markdown'], ['SRT (Subtitles)', 'srt']]) {
        const item = new PopupMenu.PopupMenuItem(label)
        item.connect('activate', () => this._exportHistory(fmt))
        this._exportSection.menu.addMenuItem(item)
      }
      this.menu.addMenuItem(this._exportSection)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      this._applyItem = new PopupMenu.PopupMenuItem(_('Apply & Restart Services'))
      this._applyItem.connect('activate', () => this._applyAndRestart())
      this.menu.addMenuItem(this._applyItem)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences...'))
      prefsItem.connect('activate', () => {
        this._extension.openPreferences()
      })
      this.menu.addMenuItem(prefsItem)
    }

    // -- Radio groups -------------------------------------------------------

    _buildRadioGroup (section, options, settingKey) {
      const current = this._settings.get_string(settingKey)
      for (const option of options) {
        const item = new PopupMenu.PopupMenuItem(option)
        if (option === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string(settingKey, option)
          section.label.set_text(`${settingKey.charAt(0).toUpperCase() + settingKey.slice(1)}: ${option}`)
          this._updateRadioOrnaments(section, option)
        })
        section.menu.addMenuItem(item)
      }
    }

    _buildLanguageGroup (section) {
      const current = this._settings.get_string('language')
      for (const [label, code] of LANGUAGES) {
        const item = new PopupMenu.PopupMenuItem(label)
        if (code === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string('language', code)
          section.label.set_text(`Language: ${label}`)
          this._updateLanguageOrnaments(section, code)
        })
        section.menu.addMenuItem(item)
      }
    }

    _updateLanguageOrnaments (section, selectedCode) {
      const items = section.menu._getMenuItems()
      for (const [i, item] of items.entries()) {
        if (item.label && i < LANGUAGES.length) {
          item.setOrnament(
            LANGUAGES[i][1] === selectedCode
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    _buildTranslateGroup (section) {
      const current = this._settings.get_string('translate-to')
      for (const [label, value] of TRANSLATE_TARGETS) {
        const item = new PopupMenu.PopupMenuItem(label)
        if (value === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string('translate-to', value)
          section.label.set_text(`Translate To: ${label}`)
          const items = section.menu._getMenuItems()
          for (const [i, menuItem] of items.entries()) {
            if (menuItem.label && i < TRANSLATE_TARGETS.length) {
              menuItem.setOrnament(
                TRANSLATE_TARGETS[i][1] === value
                  ? PopupMenu.Ornament.DOT
                  : PopupMenu.Ornament.NONE
              )
            }
          }
        })
        section.menu.addMenuItem(item)
      }
    }

    async _exportHistory (format) {
      try {
        const result = await this._client.exportHistory(format)
        if (!result) {
          Main.notify(_('Whisper NPU'), _('No history to export'))
          return
        }
        const ext = format === 'markdown' ? 'md' : format
        const homePath = GLib.get_home_dir()
        const filePath = GLib.build_filenamev([homePath, `whisper-history.${ext}`])
        const content = format === 'json' ? JSON.stringify(result, null, 2) : result
        saveToFile(filePath, content)
        Main.notify(_('Whisper NPU'), _(`History exported to ~/whisper-history.${ext}`))
      } catch (e) {
        logDebug(`Export failed: ${e.message}`)
        Main.notify(_('Whisper NPU'), _(`Export failed: ${e.message}`))
      }
    }

    _buildRecallKeyGroup (section) {
      const current = this._settings.get_string('recall-key')
      for (const key of HOTKEYS) {
        const label = this._formatHotkey(key)
        const item = new PopupMenu.PopupMenuItem(label)
        if (key === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string('recall-key', key)
          section.label.set_text(`Recall Key: ${label}`)
          const items = section.menu._getMenuItems()
          for (const [i, menuItem] of items.entries()) {
            if (menuItem.label) {
              menuItem.setOrnament(
                HOTKEYS[i] === key
                  ? PopupMenu.Ornament.DOT
                  : PopupMenu.Ornament.NONE
              )
            }
          }
        })
        section.menu.addMenuItem(item)
      }
    }

    _buildHotkeyGroup (section) {
      const current = this._settings.get_string('hotkey')
      for (const key of HOTKEYS) {
        const label = this._formatHotkey(key)
        const item = new PopupMenu.PopupMenuItem(label)
        if (key === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string('hotkey', key)
          section.label.set_text(`Hotkey: ${label}`)
          this._updateHotkeyOrnaments(section, key)
        })
        section.menu.addMenuItem(item)
      }
    }

    _updateRadioOrnaments (section, selected) {
      const items = section.menu._getMenuItems()
      for (const item of items) {
        if (item.label) {
          item.setOrnament(
            item.label.get_text() === selected
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    _updateHotkeyOrnaments (section, selectedKey) {
      const items = section.menu._getMenuItems()
      for (const [i, item] of items.entries()) {
        if (item.label) {
          item.setOrnament(
            HOTKEYS[i] === selectedKey
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    _formatHotkey (key) {
      return key.replace('KEY_', '').replace(/([A-Z])([A-Z]+)/g, (_, first, rest) => {
        return first + rest.toLowerCase()
      })
    }

    // -- Server status ------------------------------------------------------

    async _refreshStatus () {
      this._statusItem.label.set_text(_('Status: Checking...'))

      const health = await this._client.getHealth()
      if (health) {
        this._connected = true
        this._currentModel = health.model
        this._statusItem.label.set_text(_(`Status: Connected | Model: ${health.model}`))
      } else {
        this._connected = false
        this._currentModel = null
        this._statusItem.label.set_text(_('Status: Disconnected'))
      }

      this._populateModelSection()
    }

    // -- STT model management -----------------------------------------------

    _populateModelSection () {
      this._modelSection.menu.removeAll()

      const localModels = listLocalModels()

      if (localModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No models installed'), { reactive: false })
        this._modelSection.menu.addMenuItem(noModels)
      } else {
        for (const model of localModels) {
          const item = new PopupMenu.PopupMenuItem(model)
          if (model === this._currentModel) {
            item.setOrnament(PopupMenu.Ornament.DOT)
          }
          item.connect('activate', () => this._switchModel(model, item))
          this._modelSection.menu.addMenuItem(item)
        }
      }

      const available = this._cachedRemoteModels.filter(
        m => !localModels.includes(m.name)
      )

      if (available.length > 0) {
        this._modelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

        const downloadHeader = new PopupMenu.PopupMenuItem(_('Available Downloads'), { reactive: false })
        downloadHeader.label.add_style_class_name('whisper-status-label')
        this._modelSection.menu.addMenuItem(downloadHeader)

        const org = this._settings.get_string('hf-org')
        for (const model of available) {
          const downloads = model.downloads > 1000
            ? `${(model.downloads / 1000).toFixed(0)}k`
            : `${model.downloads}`
          const item = new PopupMenu.PopupMenuItem(`${model.name}  (${downloads} downloads)`)
          item.connect('activate', () => this._startDownload(org, model.name, item))
          this._modelSection.menu.addMenuItem(item)
        }
      }

      this._modelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const refreshItem = new PopupMenu.PopupMenuItem(_('Refresh List'))
      refreshItem.connect('activate', () => {
        refreshItem.label.set_text(_('Refreshing...'))
        refreshItem.reactive = false
        this._refreshModelCache()
      })
      this._modelSection.menu.addMenuItem(refreshItem)

      const activeLabel = this._currentModel || (localModels.length > 0 ? localModels[0] : '')
      if (activeLabel) {
        this._modelSection.label.set_text(_(`STT: ${activeLabel}`))
      }
    }

    async _switchModel (modelName, item) {
      if (!this._connected) {
        Main.notify(_('Whisper NPU'), _('Server is not connected'))
        return
      }

      const origLabel = item.label.get_text()
      item.label.set_text(_(`Loading ${modelName}...`))
      item.reactive = false
      this._modelSection.label.set_text(_(`Loading ${modelName}...`))

      logDebug(`Switching to model: ${modelName}`)
      const result = await this._client.setDefaultModel(modelName)

      if (result) {
        this._currentModel = modelName
        Main.notify(_('Whisper NPU'), _(`Switched to ${modelName}`))
      } else {
        item.label.set_text(origLabel)
        item.reactive = true
        Main.notify(_('Whisper NPU'), _(`Failed to load ${modelName}`))
      }
      this._populateModelSection()
    }

    // -- STT model download -------------------------------------------------

    async _refreshModelCache () {
      const org = this._settings.get_string('hf-org')
      this._cachedRemoteModels = await this._hfClient.searchModels(org) || []
      this._populateModelSection()
    }

    async _startDownload (org, modelName, item) {
      item.label.set_text(_(`Downloading ${modelName}...`))
      item.reactive = false
      this._modelSection.label.set_text(_(`Downloading ${modelName}...`))
      logDebug(`Starting download: ${org}/${modelName}`)

      try {
        await downloadModel(org, modelName)
        item.label.set_text(_(`Downloaded ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName}`))
      } catch (e) {
        logDebug(`Download failed: ${e.message}`)
        item.label.set_text(_(`Failed: ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
      this._populateModelSection()
    }

    // -- LLM model management -----------------------------------------------

    _populateLlmModelSection () {
      this._llmModelSection.menu.removeAll()

      const localModels = listLocalLlmModels()

      if (localModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No LLM models installed'), { reactive: false })
        this._llmModelSection.menu.addMenuItem(noModels)
      } else {
        for (const model of localModels) {
          const item = new PopupMenu.PopupMenuItem(model)
          if (model === this._currentLlmModel) {
            item.setOrnament(PopupMenu.Ornament.DOT)
          }
          item.connect('activate', () => this._switchLlmModel(model, item))
          this._llmModelSection.menu.addMenuItem(item)
        }
      }

      const available = this._cachedRemoteLlmModels.filter(
        m => !localModels.includes(m.name)
      )

      if (available.length > 0) {
        this._llmModelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

        const downloadHeader = new PopupMenu.PopupMenuItem(_('Available Downloads'), { reactive: false })
        downloadHeader.label.add_style_class_name('whisper-status-label')
        this._llmModelSection.menu.addMenuItem(downloadHeader)

        for (const model of available) {
          const downloads = model.downloads > 1000
            ? `${(model.downloads / 1000).toFixed(0)}k`
            : `${model.downloads}`
          const item = new PopupMenu.PopupMenuItem(`${model.name}  (${downloads} downloads)`)
          item.connect('activate', () => this._startLlmDownload(model.name, item))
          this._llmModelSection.menu.addMenuItem(item)
        }
      }

      this._llmModelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const refreshItem = new PopupMenu.PopupMenuItem(_('Refresh List'))
      refreshItem.connect('activate', () => {
        refreshItem.label.set_text(_('Refreshing...'))
        refreshItem.reactive = false
        this._refreshLlmCache()
      })
      this._llmModelSection.menu.addMenuItem(refreshItem)

      const label = this._currentLlmModel ? `LLM: ${this._currentLlmModel}` : 'Language Buddy Models'
      this._llmModelSection.label.set_text(_(label))
    }

    async _refreshLlmCache () {
      const org = this._settings.get_string('hf-org')
      this._cachedRemoteLlmModels = await this._hfClient.searchLlmModels(org) || []
      this._populateLlmModelSection()
    }

    async _switchLlmModel (modelName, item) {
      const origLabel = item.label.get_text()
      item.label.set_text(_(`Loading ${modelName}...`))
      item.reactive = false
      this._llmModelSection.label.set_text(_(`Loading ${modelName}...`))

      logDebug(`Switching LLM to: ${modelName}`)
      const result = await this._client.setLlmModel(modelName)

      if (result) {
        this._currentLlmModel = modelName
        Main.notify(_('Whisper NPU'), _(`LLM: ${modelName}`))
      } else {
        item.label.set_text(origLabel)
        item.reactive = true
        Main.notify(_('Whisper NPU'), _(`Failed to load ${modelName}`))
      }
      this._populateLlmModelSection()
    }

    async _startLlmDownload (modelName, item) {
      logDebug(`Download LLM: ${modelName}`)
      item.label.set_text(_(`Downloading ${modelName}...`))
      item.reactive = false
      this._llmModelSection.label.set_text(_(`Downloading ${modelName}...`))
      const org = this._settings.get_string('hf-org')

      try {
        await downloadLlmModel(org, modelName)
        item.label.set_text(_(`Downloaded ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName}`))
      } catch (e) {
        logDebug(`LLM download failed: ${e.message}`)
        item.label.set_text(_(`Failed: ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
      this._populateLlmModelSection()
    }

    // -- Apply settings -----------------------------------------------------

    async _applyAndRestart () {
      this._applyItem.label.set_text(_('Restarting services...'))
      this._applyItem.reactive = false

      const device = this._settings.get_string('device')
      const backend = this._settings.get_string('backend')
      const hotkey = this._settings.get_string('hotkey')
      const lang = this._settings.get_string('language')
      const recallKey = this._settings.get_string('recall-key')
      const voiceCommands = this._settings.get_boolean('voice-commands-enabled')
      const notifications = this._settings.get_boolean('notifications-enabled')
      const vadThreshold = this._settings.get_int('vad-threshold')
      const streamInterval = this._settings.get_double('stream-interval')
      const autoPunctuate = this._settings.get_boolean('auto-punctuate')
      const translateTo = this._settings.get_string('translate-to')
      const audioFeedback = this._settings.get_boolean('audio-feedback-enabled')
      const formatting = this._settings.get_boolean('dictation-formatting-enabled')

      logDebug(`Applying settings: device=${device} backend=${backend} hotkey=${hotkey} lang=${lang}`)

      try {
        await writeServiceOverride('whisper-server.service', {
          WHISPER_DEVICE: device
        })

        const pttEnv = { XDG_SESSION_TYPE: 'wayland' }
        if (lang) pttEnv.WHISPER_LANGUAGE = lang
        if (autoPunctuate) pttEnv.WHISPER_AUTO_PUNCTUATE = '1'
        if (translateTo) pttEnv.WHISPER_TRANSLATE_TO = translateTo
        if (!audioFeedback) pttEnv.WHISPER_NO_SOUND = '1'
        if (!formatting) pttEnv.WHISPER_NO_FORMATTING = '1'

        const pttArgs = ['--key', hotkey, '--backend', backend, '--recall-key', recallKey,
          '--vad-threshold', String(vadThreshold), '--stream-interval', String(streamInterval)]
        if (!voiceCommands) pttArgs.push('--no-commands')
        if (!notifications) pttArgs.push('--no-notify')

        await writePttServiceOverride(pttEnv, pttArgs)

        await restartService('whisper-server.service')
        await restartService('push-to-talk.service')

        this._applyItem.label.set_text(_('Services restarted'))
        Main.notify(_('Whisper NPU'), _('Services restarted'))
      } catch (e) {
        logDebug(`Failed to apply settings: ${e.message}`)
        this._applyItem.label.set_text(_('Restart failed'))
        Main.notify(_('Whisper NPU'), _(`Failed to restart: ${e.message}`))
      }

      this._applyItem.reactive = true
      GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 3, () => {
        this._applyItem.label.set_text(_('Apply & Restart Services'))
        return GLib.SOURCE_REMOVE
      })
    }

    // -- Cleanup ------------------------------------------------------------

    destroy () {
      if (this._dbusImpl) {
        this._dbusImpl.unexport()
        this._dbusImpl = null
      }
      if (this._overlay) {
        Main.layoutManager.removeChrome(this._overlay)
        this._overlay.destroy()
        this._overlay = null
      }
      if (this._historyOverlay) {
        Main.layoutManager.removeChrome(this._historyOverlay)
        this._historyOverlay.destroy()
        this._historyOverlay = null
      }
      if (this._settingsChangedId) {
        this._settings.disconnect(this._settingsChangedId)
        this._settingsChangedId = null
      }
      if (this._client) {
        this._client.destroy()
        this._client = null
      }
      if (this._hfClient) {
        this._hfClient.destroy()
        this._hfClient = null
      }
      this._settings = null
      super.destroy()
    }
  }
)

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

export default class WhisperNpuExtension extends Extension {
  enable () {
    this._settings = this.getSettings()
    this._indicator = new WhisperIndicator(this)
    Main.panel.addToStatusArea('whisper-npu', this._indicator)
  }

  disable () {
    if (this._indicator) {
      this._indicator.destroy()
      this._indicator = null
      this._settings = null
    }
  }
}
